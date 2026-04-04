# HippoMaps Workflow 

## 1. 这条 workflow 做什么

HippoMaps 的目标，是把单被试的海马同时放到两个维度上描述：

1. **结构维度**
   基于 `T1w/T2w` 结构像，得到海马 folded/native surface，以及海马亚区的结构分区标签。
2. **功能维度**
   基于 resting-state fMRI，得到海马 surface vertex 到皮层参考 parcel 的功能连接模式，并进一步分解为连续的功能梯度。

最终会得到两类正式结果：

1. **分析结果**
   - 结构 subfield surface label
   - hippocampal FC gradients
2. **展示结果**
   - 用 Workbench 渲染的结构图
   - 用正式流程输出的 Gradient 1 图

## 2. 输入数据

这条 workflow 只需要 3 类核心输入：

1. `T1w`
2. `T2w`
3. `resting-state fMRI`

当前项目里实际使用的是：

- 结构像：HCP 7T 的 `T1w/T2w`（`0.7 mm isotropic`）
- 功能像：HCP 7T 的体空间 resting-state `BOLD NIfTI`（`1.6 mm isotropic`）

对应的数据类型可以概括为：

```text
输入类型   内容                         常见格式
T1w       结构像                       .nii.gz
T2w       结构像                       .nii.gz
rs-fMRI   4D resting-state BOLD        .nii.gz
mask      BOLD brain mask              .nii.gz
atlas     Schaefer 皮层 atlas          .nii.gz, .dlabel.nii
```

当前正式流程里最关键的分辨率可以概括为：

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

## 3. 主流程

### Step 1. 用 HippUnfold 建立海马结构空间

首先将 `T1w/T2w`（`0.7 mm`）输入 `HippUnfold`，得到个体化的海马几何表示。

这里的 `HippUnfold` 已经包含了它内部的 `nnUNet` 结构分割步骤，所以可以把这一步整体理解为：

1. `HippUnfold` 内部先用 `nnUNet` 做海马结构分割
2. 再继续完成海马 surface 重建、边界处理和 subfield 标注

这一步会输出两类关键结构结果：

1. **海马 surface**
   - 例如 `midthickness.surf.gii`
   - 表示海马 folded/native surface 的几何形状
2. **海马结构分区 label**
   - 例如 `atlas-multihist7_subfields.label.gii`
   - 表示每个海马 vertex 属于哪个结构亚区

这里得到的结构标签从一开始就是 **surface label**。

这一步同时包含一个重要的分辨率/表示转换：

- 输入是 `0.7 mm` 结构体数据
- 输出分析 surface density 固定为 `2 mm`
- 也就是说，结构链路把高分辨率结构像转换成海马 surface 表示，并统一到当前 workflow 使用的 `2 mm` surface density

所以 `nnUNet` 属于**HippUnfold 结构链路前端**，不是功能分区算法本身。

### Step 2. 提取皮层功能参考 parcel 时序

为了刻画海马的功能连接组织，需要一个皮层侧的参考 parcel 时序。

当前项目使用 `Schaefer400` 体系，从全脑 rs-fMRI 中提取：

1. parcel-level timeseries
2. 作为附加摘要保留的 7-network mean timeseries

当前项目现在默认使用 archive 中已确认存在的 **7T CIFTI/dtseries** 作为皮层参考，因此这里恢复为 `surface/CIFTI` 参考路线。

这一阶段的典型输出是：

```text
输出                           含义
schaefer400_parcel_timeseries  parcel 时序
schaefer7_network_timeseries   7 个网络均值时序（附加摘要）
resampled atlas               重采样到 BOLD 空间的 atlas
```

这里有一个明确的分辨率转换：

- Schaefer volume atlas 原始是 `2.0 mm`
- BOLD 是 `1.6 mm`
- 因此这一步会把 atlas **从 2.0 mm 重采样到 1.6 mm BOLD 空间**

### Step 3. 把 rs-fMRI 映射到海马 surface

有了海马 surface 和全脑 BOLD 之后，下一步是把 BOLD 信号采样到海马 surface 上。

这里使用 Workbench 的 `volume-to-surface-mapping`，把体空间 rs-fMRI（`1.6 mm`）映射到海马 folded/native surface 顶点（分析 density `2 mm`），得到：

1. 每个海马 vertex 的 BOLD 时间序列
2. 左右海马各自的 surface timeseries

常见输出格式是：

```text
输出                    含义                       格式
hipp BOLD func.gii      海马 surface 时序         .func.gii
hipp BOLD numpy         同一份时序的数值版        .npy
```

这里的分辨率转换是：

- 输入信号在体空间里是 `1.6 mm`
- 输出不是体素，而是海马 `2 mm` surface density 上的 vertex timeseries

- 论文/方法学层面要求的是：**把 rs-fMRI 采样到海马 surface 上**
- Workbench -volume-to-surface-mapping 是 HippUnfold/HippoMaps 官方工程里已有的可选流程；因为我们这里输入的是 volume rs-fMRI，所以必须走这条 volume -> surface 路径

### Step 4. 重要兼容步骤：映射前几何 QC 与条件纠偏

这里有一个很重要的兼容点，应该单独说清楚。

原始 Workbench `volume-to-surface-mapping` 在某些被试和某些空间组合下，会出现一个严重问题：

1. 体数据本身不是空的
2. 但映射到 surface 后，输出的 `.shape.gii` 或相关中间结果变成**全零**
3. 后续边界检测或 label 传播会因此直接崩溃

这个缺陷的本质是：

- surface 和 volume 在 world space 上出现错位时，
- Workbench 仍然会正常返回一个结果文件，
- 但结果可能是“看上去成功、实际上全零”的假成功。

当前项目的兼容做法是：

1. 在正式执行 `volume-to-surface-mapping` 前，先用包装脚本做几何 QC
2. QC 会统一检查顶点落入 volume 的比例、直接采样的非零比例，以及采样值是否近似常数
3. 如果命中已知的负向 `x` 错位模式，就先对 surface 采样坐标做纠偏
4. 然后再执行正式的 Workbench `volume-to-surface-mapping`
5. 如果 QC 异常但不符合已知纠偏模式，就直接报错停止，而不是继续生成一个假成功结果

所以这一步不是论文方法本身，而是一个非常关键的**工程兼容修复**。没有它，某些被试会在结构后处理或 surface 映射阶段失败。

### Step 5. 计算海马的 FC 梯度

有了两部分时序之后：

1. 海马顶点时序
2. `Schaefer400` parcel 时序

就可以先计算每个海马顶点到皮层 parcels 的功能连接，再对这些 FC 模式做 `diffusion map embedding`，得到连续的功能梯度。

这一步的核心输出包括：

1. `hipp_vertex_to_parcel_fc.npy`
   每个海马 vertex 到皮层 parcels 的 FC 模式
2. `hipp_fc_gradients.npy`
   多个连续梯度分量
3. `hipp_fc_gradient1.npy`
   第一主梯度

这里得到的是 **连续标量结果**，不是离散 surface label。

### Step 6. 正式功能结果保留为 surface scalar

按论文原文口径，正式功能结果不再是离散 label，而是海马 surface 上的连续梯度。

因此当前主流程里，功能结果的正式主表示是：

1. `hipp_fc_gradients.npy`
2. `hipp_fc_gradient1.npy`
3. 正式展示图 `sub-<id>_gradient.png`

### Step 7. 当前正式主流程不再要求 volume 回投

论文主线强调的是海马 surface 上的 FC 模式与连续梯度，因此当前正式主流程不再把 `surface -> volume` 回投和 seed FC 当成必需步骤。

如果以后需要做体空间派生分析，那会被视为**额外派生分支**，不属于当前锁定的正式主流程。

### Step 8. 用 Workbench 生成正式图片

最终的正式图片包括结构图和 `Gradient 1` 图。

这里的输入是：

1. 结构 surface / structure label
2. 海马 FC gradient 标量结果

最终图像是：

1. `sub-<id>_structural.png`
2. `sub-<id>_gradient.png`

## 4. 最终到底得到什么

### Hippomaps 在哪里实际用到

这里要区分两个层面：

1. `HippUnfold`
   负责前面的海马结构分割、surface 重建和结构 subfield 标注。
2. `Hippomaps`
   在当前 workflow 里主要作为**后处理工具层**使用。

当前代码里，`Hippomaps` 最明确、最直接的使用点是：

- `hippomaps.utils.surface_to_volume`

也就是说，当前 workflow 不是“只用到了 HippUnfold”，而是：

- 前半段以 `HippUnfold` 为核心
- 后半段某些 label 回投功能明确调用了 `Hippomaps`

另外，从更宽的工作流概念上说，海马 surface 上做功能标签、再回投 volume，这整条思路本身也是 `HippoMaps` 方向的方法学范畴；只是当前项目里并不是所有步骤都直接调用 `hippomaps` 包现成函数，有一部分是本地脚本实现。

这个 workflow 最终会产出 4 类最关键的东西：

```text
类别            代表形式                          它是什么
结构 surface     .surf.gii                         海马 folded/native 几何
结构 label       .label.gii                        海马结构亚区标签
功能梯度         .npy                              海马连续 FC 梯度
功能图片         .png                              结构图与 Gradient 1 图
```

如果只问一句“最终功能结果是什么”，最准确的回答是：

**主流程最终得到的是海马 surface 上的连续 FC 梯度；不再把 volume label 当正式主结果。**

## 5. 哪些是论文步骤，哪些是本地适配

### 更接近论文核心的方法学步骤

```text
步骤                                 性质
HippUnfold（内部含 nnUNet）建结构空间  论文核心步骤
把 rs-fMRI 采样到海马 surface         论文核心步骤
海马顶点到皮层参考 parcel/network 的 FC  论文核心步骤
功能连接连续组织/梯度分析               论文核心步骤
```

### 更偏工程实现或本地适配的部分

```text
步骤/做法                                      性质
从 HCP 体空间 rs-fMRI 提取 Schaefer 参考时序    本地适配
把 FC 模式做 diffusion map 梯度分解              本地工程实现
Workbench 正式渲染流程                          工程实现
volume-to-surface 映射前几何 QC 与条件纠偏      关键兼容修复
```

## 6. 一句话介绍版

HippoMaps workflow 可以概括成一句话：

先用 `HippUnfold`（其内部包含 `nnUNet` 分割）从 `T1w/T2w` 建立个体化海马 structure surface 与 subfield label，再把 resting-state fMRI 通过官方工程里已有的 Workbench `volume -> surface` 路径映射到海马 surface，并在映射前做几何 QC 与条件纠偏；然后计算海马到 `Schaefer400` parcels 的功能连接，并做 `diffusion map embedding` 得到连续功能梯度；最后输出正式结构图和 `Gradient 1` 图。

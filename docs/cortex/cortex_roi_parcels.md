# Cortex ROI Parcels

## 1. 这是什么

这套 `parcel` 不是像 `Schaefer400` 那样预先定义好的固定 atlas parcel，而是从个体化 cortex `network` 分割里二次导出的 `network-derived connected-component parcels`。

更准确地说：

1. 先有个体化 `network` map。
2. 再把每个 `network` 在表面网格上的不连通块拆开。
3. 把面积足够大的连通块保留为 ROI/parcel。

因此这里的 `parcel` 含义是：

- `个体内` 可重复的 surface 连通块 ROI
- 适合可视化、单被试统计、方法探索
- 不等价于跨被试稳定对应的 atlas parcel

## 2. 生成原理

输入是已经存在的 cortex PFM 输出：

- `PFM_<Method>priors.dlabel.nii`
- `PFM_<Method>priors.L.label.gii`
- `PFM_<Method>priors.R.label.gii`
- 左右半球 `midthickness` surface

处理步骤如下：

1. 对每个半球，把 label map 转成每个 network 一列的二值 ROI metric。
2. 对每个 network，在 surface mesh 上找 connected components。
3. 计算每个 component 的表面积。
4. 过滤掉面积小于阈值的碎块。
5. 对保留下来的 component 按面积从大到小编号，得到 parcel 名。
6. 把这些保留 component 合并成新的 ROI label map。
7. 再从 ROI label map 提取 boundary，用于边界叠加出图。

当前默认阈值是 `25 mm^2`。

## 3. 关键脚本

```text
脚本                                  作用
scripts/cortex/derive_cortex_roi_components.py       从 network map 派生 ROI/parcel 连通块
scripts/cortex/render_cortex_roi_boundary_batch.py   批量渲染 ROI 边界叠加图
scripts/cortex/run_cortex_pfm_subject.py             提供 cortex scene 渲染与 montage 逻辑
scripts/cortex/render_cortex_pfm_scene.py            单视角 Workbench scene capture
```

## 4. 具体命令

单个 subject / method 生成 ROI parcels：

```bash
source /opt/miniconda3/bin/activate py314
python scripts/cortex/derive_cortex_roi_components.py \
  --subject 100610 \
  --method Lynch2024
```

批量生成 ROI 边界叠加图：

```bash
source /opt/miniconda3/bin/activate py314
python scripts/cortex/render_cortex_roi_boundary_batch.py \
  --subjects 100610 102311 102816 \
  --methods Lynch2024 Hermosillo2024 Kong2019 \
  --scene config/manual_wb_scenes/cortex_manual.scene \
  --roi-min-area-mm2 25
```

## 5. 输出

每个 `subject / method` 会写出：

```text
路径                                                                 含义
outputs/cortex_pfm/sub-<id>/<method>/roi_components/roi_component_stats.json   ROI 统计摘要
outputs/cortex_pfm/sub-<id>/<method>/roi_components/roi_component_stats.csv    每个 component 的面积/保留状态
outputs/cortex_pfm/sub-<id>/<method>/roi_components/hemi_L/*.label.gii         左半球 ROI parcels 与边界
outputs/cortex_pfm/sub-<id>/<method>/roi_components/hemi_R/*.label.gii         右半球 ROI parcels 与边界
outputs/cortex_pfm/sub-<id>/<method>/wb_<slug>_inflated_roi_boundaries.png     network 底图 + ROI 边界叠加图
outputs/cortex_pfm/roi_component_summary.csv                                    跨被试汇总
outputs/cortex_pfm/roi_component_summary.md                                     跨被试汇总
```

ROI 名字格式示例：

- `Default_Parietal_L_01`
- `DorsalAttention_R_02`

命名规则是：

- `network_name`
- `hemisphere`
- `该 network 内按面积排序后的序号`

## 6. 边界图是怎么画的

当前边界图不是直接替换原来的 network 图，而是在原有 network fill 上叠加 parcel 边界：

1. 先从 ROI label 生成 Workbench border。
2. 再把 border 转成可渲染的 boundary label。
3. 用同一个手调 scene 渲染 boundary 图。
4. 从 boundary 渲染图里提取边界像素。
5. 对边界做骨架化，再以细黑线叠加回原始 network 图。

这样做的目的是：

- 保留原先已经确认过的 network 颜色和布局
- 只额外提供 parcel 分界线

## 7. 解释口径

使用这套结果时，建议明确写成：

- `network-derived parcels`
- `connected-component parcels`
- `exploratory ROI parcels`

不建议直接把它写成：

- `standard atlas parcels`
- `cross-subject stable parcels`

因为不同被试、不同方法下，parcel 数量可能不同，编号也不表示跨被试一一对应。

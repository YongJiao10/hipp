# Agent Handoff

## Current Input Contract

下一个 agent 接手时，**唯一权威输入目录**是 `data/hippunfold_input`。

这套输入的含义是：
- 文件内容来自 HCP 7T 已预处理结构像与 rsfMRI
- 目录组织采用 `sub-*/anat`, `sub-*/func`，便于 `hippunfold` 直接消费
- 它是 **analysis input / hippunfold-compatible input**
- 不要把它表述成“原始 BIDS 数据”

```text
Path                  Status         Meaning
data/hippunfold_input authoritative  唯一分析输入目录
```

## Current Local-Test Outputs

当前本地测试最重要的结果目录：

```text
Path                                                                                                  Meaning
outputs/timing_runs/sub-100610_20260326_013111_local_test/wta_surface                                 当前单被试 WTA 结果
outputs/timing_runs/sub-100610_20260326_013111_local_test/wta_surface/wta_surface_summary.json        左右侧和合并比例
outputs/timing_runs/sub-100610_20260326_013111_local_test/wta_surface/sub-100610_hipp_wta_native.png 原生 surface 图
outputs/timing_runs/sub-100610_20260326_013111_local_test/wta_surface/sub-100610_hipp_wta_overlay_native.png ASHS 风格 overlay 图
```

## Current Scientific Status

- 当前 WTA 结果是 **local-test surface-based result**
- 已经可以回答“此人海马可分成哪些功能网络、各占多少比例”
- 严格的 HPC 正式版仍应继续完成：
  - 完整 HippUnfold 正式运行
  - 体积回投到最终分析空间
  - HPC 上的 seed FC

## Terminology To Preserve

- 用 `analysis input`、`hippunfold-compatible input`
- 不要再用 `raw data` 指代项目输入目录
- 不要再用 `bids staging` 指代当前唯一输入目录

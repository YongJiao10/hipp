# Multi-Dataset & Multi-Cohort Normalized WTA Long-Axis Stacked Bar Overview (2-Column Layout)

本文档说明跨数据集海马 2mm Normalized WTA 图谱与长轴（Anterior-to-Posterior）网络分布的 **双大列（2-Column）** 可视化方案及产物。

## Additive HCP/JJ_TMS 7-network comparison

## Signal branch contract

The legacy renderer keeps all four resolution/preprocessing grid cells. The
maintained comparison renderer additionally accepts an explicit 2mm/36p
signal branch: standard or ctxreg. Every renderer-owned cohort entry
materializes that branch before loading data, and every WTA manifest must
record the same branch, except pre-signal_branch HCP Kong manifests, which
are accepted only under the explicit named legacy provenance contract
(formal routea_ctxreg signal roots with the standard reference root and
fully ctxreg consumed inputs). Any other missing or mismatched ctxreg
declaration stops the render.

The exact WTA label array, label map, surface, template spec, and GIFTI label
are read from manifest-declared paths. The NPY and GIFTI labels must have the
same shape and values. Surface cache directories and cache manifests include
the signal branch, so Standard and CtxReg panels cannot reuse one another's
products. The ctxreg WTA path is
work/group_atlas/routea_ctxreg[/cohort]/wta-top/kong2022/...; no extra
wta-top level is valid.
The `hcp_jjtms_7net` branch adds Kong2022 7-Network and Lynch2024 7-Network method columns while preserving the original WTA layout, dataset grouping, panel styling, and Workbench rendering. The fixed common roster is HCP 7T `N=151`, Iowa HC `N=67`, Iowa Patient `N=26` (sub-3003 excluded), Pure MDD `N=7`, and Comorbid MDD `N=19`; Stanford is retained only by the legacy branch.

---

## 1. 布局设计与队列分布

本可视化采用 **左右两大列 (2-Column Grid, 16:9 宽屏架构，共 10 个队列，N = 528)** 进行对比排版：

### 左列 (Column 1 - 5 个对照与基准队列, N = 336)

1. **HCP 7T** (Healthy Young Adults, $N=151$): 高分辨率青年健康基准
2. **Iowa 3T HC** (Healthy Controls, $N=67$): 临床对照组
3. **Iowa 3T Patient** (Depression Patients, $N=26$, sub-3003 excluded): 重度抑郁症患者队列
4. **Stanford NTHC** (Non-Trauma Healthy Controls, $N=51$): 斯坦福无创伤健康对照
5. **Stanford TEHC** (Trauma-Exposed Healthy Controls, $N=40$): 斯坦福创伤暴露健康对照

### 右列 (Column 2 - 5 个临床与亚组队列, N = 192)

1. **Stanford NTS** (Non-Trauma Symptomatic, $N=56$): 斯坦福无创伤症状组总队列
2. **Stanford NTS Any MDD** (Non-Trauma MDD [All], $N=43$): 无创伤抑郁亚组（包含合并症）
3. **Stanford NTS Pure MDD** (Non-Trauma Pure MDD, $N=11$): 单纯抑郁亚组（无其他共病）
4. **Stanford TIS** (Trauma-Induced Symptomatic, $N=47$): 斯坦福创伤诱导症状组
5. **Stanford TIS PTSD** (Trauma-Informed PTSD, $N=35$): 斯坦福创伤性应激障碍亚组

---

## 2. 架构保障机制（Single Source of Truth & Data Integrity Gate）

- **单源真理原则（SSOT）**：彻底废除 Overview 静态截图截取方式，统一直接根据各队列 `group_root` 下的 `group_atlas_manifest.json` 解析 Canonical GIITI 标签文件，直接调用 Workbench 进行 4 视角图谱渲染，保证 3D 脑图视觉与右侧 20-Bin Stacked Bar 统计数据 100% 来源统一。
- **运行时一致性门禁（Data Integrity Gate）**：在数据加载时，强制比对底层 NPY 数组与 GIITI 标签文件的逐顶点一致性，从源头杜绝不同分析分支（如 `fisherz` vs `weighted_fisherz`）的数据覆盖或混淆。
- **Canonical label fail-closed**：长轴统计只读取 `group_functional_atlas_labels.npy` 并与 manifest 指向的 GIFTI 逐顶点核对；不会把 `labels_count` 或其他旧 variant 静默当作当前 WTA。
- **路径与 assignment 单一入口**：活跃默认路径由 `scripts/visualize/lib/multidataset_cohorts.py` 统一解析；Kong2022 使用 canonical7 的 `normalized_top20_count` WTA。canonical8 仅能通过显式 resolver 选择历史归档路径，不作为默认输入。

Canonical7 的维护输出文件名使用无后缀网格标签，例如
`multidataset_wta_normalized_longaxis_stacked_2col_2mm_22p.png`；旧的

---

## 3. 核心文件与产物

| 文件名 | 描述 |
| :--- | :--- |
| [`render_multidataset_wta_normalized_longaxis_stacked_2col.py`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/render_multidataset_wta_normalized_longaxis_stacked_2col.py) | 2列布局渲染生成主脚本（含 SSOT 实时渲染与一致性门禁） |
| [`multidataset_wta_normalized_longaxis_stacked_2col.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/multidataset_wta_normalized_longaxis_stacked_2col.png) | 高分辨率渲染图谱 ($46 \times 27\ \text{in}$, 220 DPI) |
| [`multidataset_wta_normalized_longaxis_bins_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/multidataset_wta_normalized_longaxis_bins_stats.tsv) | 10 个队列左右半球 20 个 Bin 的网络百分比统计表 |
| [`render_multidataset_wta_normalized_longaxis_stacked_2col_backup_v1.py`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/render_multidataset_wta_normalized_longaxis_stacked_2col_backup_v1.py) | 历史版本代码备份 |

---

## 4. 运行与复现指令

根据 Repository Agent 运行时环境规范，使用 `py314` 环境执行：

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures wta_normalized_longaxis_stacked_2col

# 双分支模式：2mm/36p standard + ctxreg（分支限定名 *_standard.png / *_ctxreg.png）
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --figures wta_normalized_longaxis_stacked_2col --signal-branch all
```

支持自定义输出路径参数：

- `--out <path_to_png>`: 指定输出 PNG 图片路径
- `--tsv <path_to_tsv>`: 指定输出 TSV 统计数据路径

---

## 5. 性能与面板缓存机制（`panels/`）

为避免在微调排版或统计展示时重复调用 Connectome Workbench 进行高开销的离屏 3D 渲染，脚本将各队列的海马表面图缓存到 `panels/<cohort>_wta_surface/`。

每个 `*_wta_biglegend.png` 都配有同名的 `.cache.json` sidecar，记录左右半球 group-atlas manifest、canonical surface/label GIFTI、spec、scene 等输入的路径、mtime、大小和 SHA-256，以及输出 PNG 的校验信息。只有输入 provenance 和输出 fingerprint 完全匹配时才复用缓存；缺少 sidecar、输入被刷新或输出被修改都会自动重绘。因此 Kong2022 的 `normalized_top20_count` 与 Lynch2024 的 `fisherz` 不会因旧 PNG 残留而串用。

首次重绘会调用 Workbench，后续输入未变化时直接复用已验证缓存。

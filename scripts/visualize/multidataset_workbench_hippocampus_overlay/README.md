# Cross-Dataset Hippocampal Workbench Parcellation Overlay & Subregion Vertex Composition (2-Column Landscape Layout)

本文档记录跨数据集海马功能图谱（$K=3$ RouteA Fisher-z）的 3D Workbench 原生表面图谱渲染与亚区顶点构成统计（12 群组双列宽屏布局，包含 2mm/4mm 与 36p 变体）。

## Additive HCP/JJ_TMS 7-network comparison

## Signal branch contract

The legacy renderer uses the two 36p grid cells. The maintained comparison
renderer can be selected with an explicit 2mm/36p signal branch, standard or
ctxreg. Each cohort entry and each panel cache carries that branch. A ctxreg
render requires both hemisphere manifests to record signal_branch: ctxreg,
except pre-signal_branch HCP Kong manifests, which are accepted only under
the explicit named legacy provenance contract (formal routea_ctxreg signal
roots with the standard reference root and fully ctxreg consumed inputs).
Any other missing declaration is an error, not a Standard substitute.

The overlay consumes only the exact Fisher-z assignment files declared by
each hemisphere manifest and the canonical group_labels_final.npy. The
manifest-declared label assets are validated before BrainSpace rendering, and
the branch is part of the panel provenance fingerprint and filename. This
keeps the PNG, TSV, cache sidecar, and manifest-selected labels on the same
signal branch.

The active renderer entry point requires signal_branch explicitly; the
historical legacy wrapper remains independently maintained and is not changed
by this branch-aware implementation.
The `hcp_jjtms_7net` branch adds Kong2022 7-Network and Lynch2024 7-Network method columns while retaining the BrainSpace surface panels, dataset bands, side labels, and layout. It uses the fixed common roster HCP 7T `N=151`, Iowa HC `N=67`, Iowa Patient `N=26` (sub-3003 excluded), Pure MDD `N=7`, and Comorbid MDD `N=19`; Stanford is excluded from this comparison branch only. For the 2mm/36p standard and ctxreg cells, the suite additionally passes the WTA `--tm-contrast-manifest`, which renders a third template-matching column (TM Hippo Atlas + its left/right 512-vertex composition card for Anterior/VAN, Middle/SMN, and Posterior/DMN). The overlay does not recompute or substitute the TM atlas image: it consumes the exact WTA TM atlas PNG and its declared TM label GIFTIs from `multidataset_hipp_cortex_gradient_wta/panels/`, derives the vertex counts from those labels, and fails closed if an exact source is missing. HCP 7T has no TM dominant atlas and renders blank in that column; Lynch2024 has no TM atlas and is never substituted — the TM column follows the Kong2022 cohort roster.

---

## 1. 布局结构与群组划分

图表采用标准 **2 列 × 6 行** 宽屏横版排版（220 DPI，约 `10350 × 7258` 像素）：

- **左列（Column 1: Controls, Clinical & MDD Subgroups, 6 cohorts, N=362）**：
  1. **HCP 7T** (Healthy Young Adults, N=151)
  2. **Iowa 3T HC** (Healthy Controls, N=67)
  3. **Iowa 3T Patient** (Depression Patients, N=26; sub-3003 excluded)
  4. **Iowa 3T Pure MDD** (Pure MDD Subgroup, N=7)
  5. **Iowa 3T Comorbid MDD** (Comorbid MDD Subgroup, N=19)
  6. **Stanford NTHC** (Non-Trauma Healthy Controls, N=51)

- **右列（Column 2: Stanford Trauma & Clinical Subgroups, 6 cohorts, N=192）**：
  1. **Stanford TEHC** (Trauma-Exposed Controls, N=40)
  2. **Stanford NTS** (Non-Trauma Symptomatic, N=56)
  3. **Stanford NTS Any MDD** (Non-Trauma MDD, N=43)
  4. **Stanford NTS Pure MDD** (Non-Trauma Pure MDD, N=11)
  5. **Stanford TIS** (Trauma-Exposed Symptomatic, N=47)
  6. **Stanford TIS PTSD** (Trauma-Informed PTSD, N=35)

---

## 2. 核心视觉要素与字号规范

每个 Cohort Row 包含 3 个子卡片，全图字号针对 48 英寸宽屏进行深度优化：

1. **Cohort Profile**：数据集徽标（21pt）、群组名称（21–26pt heavy）、样本量 $(N)$（19pt bold）与临床描述（13–14.5pt italic）。
2. **3D Hippocampus Atlas**：BrainSpace 1x2 原生视角（Lateral & Medial）表面图谱（等比例高精渲染）。
3. **Subregion Vertex Composition & Networks (L & R)**：
   - 左右半球各自独立的标题栏（19.5pt heavy）。
   - 顶点比例堆叠横向条形图（Anterior / Middle / Posterior，标签文字 15.5pt heavy，带黑色轮廓描边）。
   - 各亚区的精确顶点计数（以 `den-512` 为基准）与百分比占比（17.5pt heavy）。
   - 各亚区分配的主导皮层功能网络及 Fisher-z 连接强度值（16.5pt heavy，匹配网络主题色）。

底部 Footer 汇总：

- 全群组左右侧海马亚区平均顶点数与标准差（16.5pt bold）。
- 规范化网络图例（15pt）。

---

## 3. 产物清单

| 文件类型 | 文件路径 | 说明 |
| :--- | :--- | :--- |
| **2mm 36P 主图 (220 DPI)** | [`multidataset_workbench_hippocampus_overlay_2mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_workbench_hippocampus_overlay/multidataset_workbench_hippocampus_overlay_2mm_36p.png) | 2 列 12 群组 2mm 36P 高清总览图（清晰大字号） |
| **2mm 36P 统计表 (TSV)** | [`stats/multidataset_workbench_hippocampus_overlay_2mm_36p_vertex_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_workbench_hippocampus_overlay/stats/multidataset_workbench_hippocampus_overlay_2mm_36p_vertex_stats.tsv) | 12 群组左右半球三亚区顶点数与连接网络全量统计 |
| **4mm 36P 主图 (220 DPI)** | [`multidataset_workbench_hippocampus_overlay_4mm_36p.png`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_workbench_hippocampus_overlay/multidataset_workbench_hippocampus_overlay_4mm_36p.png) | 2 列 12 群组 4mm 36P 高清总览图（清晰大字号） |
| **4mm 36P 统计表 (TSV)** | [`stats/multidataset_workbench_hippocampus_overlay_4mm_36p_vertex_stats.tsv`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_workbench_hippocampus_overlay/stats/multidataset_workbench_hippocampus_overlay_4mm_36p_vertex_stats.tsv) | 12 群组左右半球三亚区顶点数与连接网络全量统计 |
| **渲染脚本** | [`render_multidataset_workbench_hippocampus_overlay.py`](file:///home/yojiao/hpchome/hipp/scripts/visualize/multidataset_workbench_hippocampus_overlay/render_multidataset_workbench_hippocampus_overlay.py) | 自动化提取、BrainSpace 渲染与排版整合脚本（内置 `panels/` 缓存） |

---

## 4. 运行与复现指令

小图缓存必须通过 `.sources.json` 中源标签、表面、manifest、场景、绘图代码、
网格和输出内容哈希校验；旧的仅按文件大小复用方式已移除。缺少源文件直接报错。

```bash
# 独立入口已移除；统一刷新 2mm/4mm 的 36p 两图及统计表
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures workbench_hippocampus_overlay

# 双分支模式：2mm/36p standard + ctxreg（分支限定名 *_standard.png / *_ctxreg.png）
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --figures workbench_hippocampus_overlay --signal-branch all
```

## 5. Exact artifact contract

For each hemisphere, the renderer consumes the exact files declared by the
manifest's assignment_variants.fisherz entry:

- cluster_assignment_status
- group_functional_atlas_labels
- group_functional_atlas_label_map
- template_functional_atlas_label

The declared assignment files must exist inside their hemisphere run
directory. group_labels_final.npy is the only accepted cluster-label array;
group_labels.npy is not a compatibility substitute. The status JSON must
declare assignment_metric: "top1_fisher_z" and complete records for cluster
IDs 1, 2, and 3. The renderer also requires manifest-declared surfaces/spec and
the canonical Workbench scene to be regular files. These exact paths are
included in each panel cache fingerprint so stale panels cannot silently be
reused. Before rendering, the formal gradient-assignment validator additionally
checks the cluster/functional-label NPY-GIFTI agreement, label map, profile,
and status semantics.

The retained formal products remain the main branch's 2mm/4mm × 36p
grids and the additive hcp_jjtms_7net comparison outputs. The active
comparison loader now supports the explicitly selected standard and ctxreg
branches; it resolves formal branch roots and validates their manifests
before rendering. The historical legacy wrapper remains independently
managed. The unified suite entry point defaults to its 36p grid
matrix; pass --signal-branch ctxreg (ctxreg cell only) or --signal-branch all
(36p standard grids plus the branch-qualified ctxreg cell) for branch-pair
products; the branch-aware comparison renderer must
be invoked with its explicit branch option. No
reference-worktree PNG, manifest, CSV, or TSV is an input to the formal
renderer. An explicitly declared signal branch other than standard or ctxreg
is rejected closed.

# Handoff: LSS esTT — HippUnfold 重跑 + Cortex PFM + Cluster FC + Individual (Spectral, Gradient, WTA-top)

**更新时间：** 2026-05-28 ~21:10 UTC  
**当前状态：** 
- ✅ 36P & 24P Group-level 核心流程均已 100% 完成
  - ✅ 36P/24P HippUnfold 重跑 (T1w only) 25/25 完成
  - ✅ 36P/24P Step B (海马 surface 采样) 25/25 完成
  - ✅ 36P/24P Step C (Cortex PFM) 25/25 完成
  - ✅ 36P/24P Step D (Cluster FC) 25/25 × 6 combos 完成
- ✅ 36P Individual-level Hippocampal **Network-Spectral** Parcellation 已 100% 完成并通过验证
  - ✅ 23名 QC-pass 受试者 × 2侧 (L, R) × 2平滑度 (2mm, 4mm) = 92个 combos 均 100% 完成，无缺失文件，无特征形状错误，且未分配脑区比例为 0.0%
- ✅ 36P Individual-level Hippocampal **Gradient-Spectral** Parcellation 已 100% 完成并通过验证
  - ✅ 23名 QC-pass 受试者 × 2侧 (L, R) × 2平滑度 (2mm, 4mm) = 92个 combos 均 100% 完成，0 缺失文件，0 形状错误，1/276 cluster 无正 ROI 连接 (sub-625, 4mm, hemi-L, cluster 1)
  - ✅ 算法：BrainSpace `GradientMaps(n_components=5, kernel="pearson")` → G1,G2,G3 扩散嵌入特征 → Cosine 亲和度 ⊙ 海马空间邻接 → `SpectralClustering` K=3 固定
  - ✅ 脚本：`scripts/run_individual_kong2022_gradient_spectral.py` + `scripts/run_estt_individual_gradient_spectral_batch.py`
- ✅ 36P Individual-level Hippocampal **WTA-top (Winner-Take-All)** Network Assignment 已 100% 完成并通过验证
  - ✅ 23名 QC-pass 受试者 × 2侧 (L, R) × 2平滑度 (2mm, 4mm) = 92个 combos 均 100% 完成，0 缺失文件，0 形状错误，未分配顶点比例仅为 0.08% (40/47104 vertices)
  - ✅ 算法：直连皮层网络投票（Winner-Take-All Top-20）
  - ✅ 脚本：`scripts/run_individual_kong2022_wta_top.py` + `scripts/run_estt_individual_wta_top_batch.py`


---

## 路径约定

| 变量 | 路径 |
|------|------|
| LSS 根 | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/` |
| 36P 工作目录 | `{LSS根}/DataOrg_PFM_36P/hipp_outputs/` |
| 24P 工作目录 | `{LSS根}/DataOrg_PFM_24P/hipp_outputs/` |
| 本地脚本 | `{LSS根}/src/hipp/` |
| hipp repo | `/home/yojiao/hpchome/hipp/`（脚本/资产来源，不写入结果） |
| 个体化谱聚类目录 | `{36P工作目录}/estt_kong2022_areal_individual_spectral/`（隔离的个体级别工作目录） |
| 个体化 Network-Spectral 结果 | `{个体化谱聚类目录}/outputs/sub-<ID>/group-network-spectral/` |
| 个体化 Gradient-Spectral 结果 | `{个体化谱聚类目录}/outputs/sub-<ID>/group-gradient-spectral/` |
| 个体化 WTA-top 结果 | `{个体化谱聚类目录}/outputs/sub-<ID>/wta-top/` |

---

## 核心流程状态与输出审计

### 1. Group-level 流程状态
*   **HippUnfold 表面重建 (T1w only)**：✅ 25/25人全部完成。包括最后一个之前卡住的受试者 `sub-846` 已完成（363/363 steps done）。
*   **Step A：QC 质控评估**：✅ 已完成重新评估并更新了 QC 报告（[QC报告](file:///home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/qc/hippunfold_qc_lss_esTT_25subjects.md)）。
    *   **结论**：QC 通过 23 人，排除 2 人（`sub-677` Dice-L = 0.108 失败，`sub-778` Dice-R = 0.625 失败且 24P null-fMRI 超过阈值）。
    *   **名单 (23人)**：`369 372 376 404 405 409 413 416 418 423 430 483 532 585 610 625 634 672 720 728 756 760 846`
*   **Step B：海马 surface 功能时序采样**：✅ 36P 和 24P 均已 100%（25/25）完成。
*   **Step C：Cortex PFM + 网络时序提取**：✅ 36P 和 24P 均已 100%（25/25）完成。
*   **Step D：Cluster FC 矩阵计算**：✅ 36P 和 24P 均已 100%（25/25）完成，输出 6 个 atlas combos 的全部相关性 FC 矩阵。

### 2. Individual-level 流程状态 (eSTT Kong2022-Areal Individual Spectral)
该目录包含了隔离的、单受试者级别的海马网络谱聚类管道及输出。
*   **当前状态**：**已对 23 名 QC-pass 受试者完成全部 92 个 combos 的计算并验证通过**。
    *   验证指标：`n_missing_files` = 0, `n_shape_errors` = 0, `unassigned_ratio` = 0.0。
*   **主要脚本**：
    *   `scripts/extract_cortex_kong2022_areal_mshbm_roi_timeseries.py`
    *   `scripts/materialize_vertex_to_roi_fisherz.py`
    *   `scripts/run_individual_kong2022_network_spectral.py` (单受试者谱聚类核心脚本)
    *   `scripts/run_estt_individual_spectral_batch.py` (批处理与验证脚本)
*   **算法原理**：
    1.  **个体化时序桥接**：提取个体化皮层网络分区 (MSHBM) 对应的 ROI 时序。
    2.  **FC 矩阵生成**：计算海马顶点到皮层个体化脑区的 Fisher-z 转换相关系数（512 × 400）。
    3.  **海马谱聚类**：将顶点间 FC 矩阵的余弦相似度作为特征，结合海马表面的空间邻接矩阵（Spatial Adjacency Matrix）约束，进行谱聚类。
    4.  **K 选择与重排**：在单受试者水平利用无冗余约束算法选择最优 K 值（K=2..10），并利用海马长轴三维坐标点集的特征向量，对聚类后的 Clusters 进行长轴重排。
    5.  **皮层网络映射**：在个体水平对每个海马 Cluster 进行 Top20 ROI 网络投影（多数投票）并输出映射结果。
*   **单受试者输出目录契约**：
    *   路径：`outputs/sub-<ID>/group-network-spectral/kong2022/<2mm|4mm>/hemi_<L|R>/`
    *   预期文件：
        *   `vertex_to_roi_fisherz.npy` / `subject_feature_after_top10.npy` (FC 特征)
        *   `cluster_labels.npy` / `cluster_label.gii` (分类标签)
        *   `k_selection_summary.json` / `k_selection_summary.tsv` (K 选择总结)
        *   `cluster_assignment_status.json` / `cluster_roi_profile.npy` / `cluster_network_profile.npy` (脑区与网络映射剖面)
        *   `manifest.json` (完整性及元数据描述)

### 3. Individual-level 流程状态 (eSTT Kong2022-Areal Individual Gradient-Spectral)
该目录包含了隔离的、单受试者级别的海马**梯度**谱聚类管道及输出，采用与本地 `gradient-spectral` 相同的 BrainSpace 扩散嵌入方法。
*   **当前状态**：**已对 23 名 QC-pass 受试者完成全部 92 个 combos 的计算并验证通过**。
    *   验证指标：`n_missing_files` = 0, `n_shape_errors` = 0, `unassigned_ratio` = 0.0036 (1/276 cluster, sub-625 4mm hemi-L cluster 1)。
*   **主要脚本**：
    *   `scripts/gradient_adapter.py` (BrainSpace GradientMaps 适配器，从 hipp repo 提取)
    *   `scripts/run_individual_kong2022_gradient_spectral.py` (单受试者梯度谱聚类核心脚本)
    *   `scripts/run_estt_individual_gradient_spectral_batch.py` (批处理与验证脚本)
*   **算法原理**：
    1.  **输入复用**：直接使用已生成的 `vertex_to_roi_fisherz.npy` (512 × 400)，无需重新提取时序。
    2.  **Column z-score 标准化**：对每列 (ROI) 在顶点间做 z-score 标准化。
    3.  **BrainSpace 扩散嵌入**：`GradientMaps(n_components=5, kernel="pearson")` → 连续功能梯度 (G1..G5)。
    4.  **特征选择**：取 G1, G2, G3 作为聚类特征 (3 维梯度特征)。
    5.  **谱聚类**：梯度特征 Cosine 亲和度 ⊙ 海马表面邻接矩阵 (Hadamard 积) → `SpectralClustering`，**K=3 固定** (与本地 group-level gradient-spectral 一致)。
    6.  **长轴重排**：利用海马长轴 (SVD) 对聚类标签排序。
    7.  **皮层网络映射**：对每个海马 Cluster 进行 Top-20 ROI 网络投影（多数投票）。
*   **单受试者输出目录契约**：
    *   路径：`outputs/sub-<ID>/group-gradient-spectral/kong2022/<2mm|4mm>/hemi_<L|R>/`
    *   预期文件：
        *   `vertex_to_roi_fisherz.npy` (512, 400) — 原始 Fisher-z FC
        *   `hippomaps_gradients_full.npy` (512, 5) — 所有 5 个梯度分量
        *   `hippomaps_lambdas.npy` (5,) — 扩散嵌入特征值
        *   `hippomaps_ev_fractions.npy` (5,) — 解释方差比例
        *   `cluster_labels.npy` / `cluster_label.gii` (分类标签, K=3)
        *   `k_selection_summary.json` / `k_selection_summary.tsv` (K 选择总结, mode=force_final_k)
        *   `cluster_assignment_status.json` / `cluster_roi_profile.npy` / `cluster_network_profile.npy` (脑区与网络映射剖面)
        *   `manifest.json` (完整性及元数据描述)

### 4. Individual-level 流程状态 (eSTT Kong2022-Areal Individual WTA-top)
该目录包含了隔离的、单受试者级别的海马**Winner-Take-All (WTA)**网络分配管道及输出，采用与本地组级 `wta-top` 相同的皮层 Top-20 ROI 多数投票算法。
*   **当前状态**：**已对 23 名 QC-pass 受试者完成全部 92 个 combos 的计算并验证通过**。
    *   验证指标：`n_missing_files` = 0, `n_shape_errors` = 0, 未分配顶点比例仅为 0.08%（92个 combos 中共计 47104 个顶点中仅 40 个未分配）。
*   **主要脚本**：
    *   `scripts/run_individual_kong2022_wta_top.py` (单受试者 WTA 多数投票核心脚本)
    *   `scripts/run_estt_individual_wta_top_batch.py` (批处理与验证脚本)
*   **算法原理**：
    1.  **输入复用**：直接使用已生成的 `vertex_to_roi_fisherz.npy` (512 × 400)。
    2.  **正相关筛选**：对每个海马顶点，保留其到皮层 400 个 ROI 中具有有限且 strictly positive (`> 0`) 的 Fisher-z FC 值。若无正相关连接，则标记为 unassigned (标签 9)。
    3.  **Top-20 排序**：按 Fisher-z 相关系数大小对正相关 ROIs 进行降序排列，筛选出 Top-20。
    4.  **皮层网络多数投票**：获取 Top-20 ROIs 所属的 canonical 8 大网络分类，以网络出现次数（频数）决定该顶点所属的网络分类。如果出现频数相同，则通过 Fisher-z 关联强度的总和（Sum Score）打破平局。
    5.  **空间与颜色映射**：将优胜的网络类别赋予该顶点，并根据 `cross_atlas_network_merge.json` 约定的 8 个 canonical 经典网络进行颜色映射输出。
*   **单受试者输出目录契约**：
    *   路径：`outputs/sub-<ID>/wta-top/kong2022/<2mm|4mm>/hemi_<L|R>/`
    *   预期文件：
        *   `vertex_to_roi_fisherz.npy` (512, 400) — 原始 Fisher-z FC
        *   `cluster_labels.npy` / `cluster_label.gii` (Winner-Take-All 顶点级分类标签与 dlabel GIFTI，标签 1-8 为网络，9 为 unassigned)
        *   `k_selection_summary.json` / `k_selection_summary.tsv` (K 选择总结，mode=clusterless)
        *   `cluster_assignment_status.json` (顶点投票明细、频数与分数 margin 信息)
        *   `manifest.json` (完整性及元数据描述)

---


## 环境信息与下游 Python 配置

| 项目 | 值 |
|------|-----|
| HippUnfold 环境 | `hippo2`（CLI 2.0.0）|
| Downstream Python env | `/home/yojiao/.conda/envs/py314/bin/python` |
| 依赖包校验 | `nibabel`, `numpy`, `scipy`, `scikit-learn` 均已安装且测试通过 |

---

## 后续分析建议
1.  **受试者过滤**：在后续分析中，应排除 `sub-677` 和 `sub-778`。个体化聚类目录下的 `outputs/` 已经移除了这两名受试者，仅保留了 23 个 QC-pass 受试者的数据。
2.  **个体聚类与组聚类的联合分析**：可将组水平谱聚类模板 `group-intrinsic-spectral` 的海马划分结果，与本目录生成的个体化谱聚类 `cluster_label.gii` 结果进行重叠度对比，以分析海马功能分区在个体水平上的异质性和特异性。

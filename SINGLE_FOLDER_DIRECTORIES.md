# 4 个 Outputs 目录单 Folder 目录层级汇总报告

> 本文档**仅包含**项目四个 `outputs_*` 目录（`outputs_iowa_estt`, `outputs_iowa_jj_tms`, `outputs_stanford`, `outputs_hcp`）中的非归档单 Folder 目录层级记录。

---

## 1. outputs_iowa_estt

- **单 Folder 目录链数量**: `1` 条

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_iowa_estt/inputs` | `outputs_iowa_estt/inputs` -> `outputs_iowa_estt/inputs/iowa` | 2 |

---

## 2. outputs_iowa_jj_tms

- **单 Folder 目录链数量**: `144` 条（包含 `3` 条固定结构链与 `141` 条受试者批处理重复链）

### 2.1 固定结构层级链

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_iowa_jj_tms/inputs` | `outputs_iowa_jj_tms/inputs` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch` | 4 |
| 2 | `outputs_iowa_jj_tms/reruns` | `outputs_iowa_jj_tms/reruns` -> `outputs_iowa_jj_tms/reruns/pfm_nsi_v0p1p0_sparse0p25_single_run01` | 2 |
| 3 | `outputs_iowa_jj_tms/inputs/.../_batch_logs` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/_batch_logs/jj_tms_hippunfold_full_corobl` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/_batch_logs/jj_tms_hippunfold_full_corobl/20260809T065738Z` | 2 |

### 2.2 受试者批处理重复链（示例前 10 条）

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_iowa_jj_tms/.../sub-1001` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1001` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1001/hippunfold` | 2 |
| 2 | `outputs_iowa_jj_tms/.../sub-1002` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1002` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1002/hippunfold` | 2 |
| 3 | `outputs_iowa_jj_tms/.../sub-1003` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1003` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1003/hippunfold` | 2 |
| 4 | `outputs_iowa_jj_tms/.../sub-1004` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1004` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1004/hippunfold` | 2 |
| 5 | `outputs_iowa_jj_tms/.../sub-1005` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1005` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1005/hippunfold` | 2 |
| 6 | `outputs_iowa_jj_tms/.../sub-1006` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1006` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1006/hippunfold` | 2 |
| 7 | `outputs_iowa_jj_tms/.../sub-1007` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1007` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1007/hippunfold` | 2 |
| 8 | `outputs_iowa_jj_tms/.../sub-1008` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1008` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1008/hippunfold` | 2 |
| 9 | `outputs_iowa_jj_tms/.../sub-1009` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1009` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1009/hippunfold` | 2 |
| 10 | `outputs_iowa_jj_tms/.../sub-1010` | `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1010` -> `outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1010/hippunfold` | 2 |

---

## 3. outputs_stanford

- **单 Folder 目录链数量**: `85` 条（包含 `2` 条核心结构入口链与 `83` 条模板/可视化层级链）

### 3.1 核心结构与入口链

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_stanford/inputs` | `outputs_stanford/inputs` -> `outputs_stanford/inputs/stanford` | 2 |
| 2 | `outputs_stanford/reruns` | `outputs_stanford/reruns` -> `outputs_stanford/reruns/shifted_cosine_crossfit_20260711` | 2 |

### 3.2 图谱模板与可视化层级链示例

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_stanford/reruns/.../gradient-spectral` | `outputs_stanford/reruns/shifted_cosine_crossfit_20260711/group_atlas/routea_symp_full_k3_fisherz_k3only/gradient-spectral` -> `outputs_stanford/reruns/shifted_cosine_crossfit_20260711/group_atlas/routea_symp_full_k3_fisherz_k3only/gradient-spectral/kong2022` | 2 |
| 2 | `outputs_stanford/reruns/.../present` | `outputs_stanford/reruns/shifted_cosine_crossfit_20260711/present` -> `outputs_stanford/reruns/shifted_cosine_crossfit_20260711/present/overviews` | 2 |
| 3 | `outputs_stanford/work/.../hemi_L/template` | `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3/wta-top/kong2022/4mm/cov_0p80/hemi_L/template` -> `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3/wta-top/kong2022/4mm/cov_0p80/hemi_L/template/workbench_assets` | 2 |
| 4 | `outputs_stanford/work/.../hemi_R/template` | `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3/wta-top/kong2022/4mm/cov_0p80/hemi_R/template` -> `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3/wta-top/kong2022/4mm/cov_0p80/hemi_R/template/workbench_assets` | 2 |

---

## 4. outputs_hcp

- **单 Folder 目录链数量**: `3910` 条（包含 Fold/受试者 交叉验证重复结构层级）

### 4.1 Fold 交叉验证重复链示例（前 10 条示例）

| 序号 | 起始层级路径 | 完整单 Folder 嵌套链 | 层级深度 |
| :--- | :--- | :--- | :---: |
| 1 | `outputs_hcp/.../fold-01/.../hemi_L/template` | `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-01/group_atlas/routea_fold01_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template` -> `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-01/group_atlas/routea_fold01_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template/workbench_assets` | 2 |
| 2 | `outputs_hcp/.../fold-01/.../hemi_R/template` | `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-01/group_atlas/routea_fold01_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/template` -> `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-01/group_atlas/routea_fold01_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/template/workbench_assets` | 2 |
| 3 | `outputs_hcp/.../fold-02/.../hemi_L/template` | `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-02/group_atlas/routea_fold02_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template` -> `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-02/group_atlas/routea_fold02_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template/workbench_assets` | 2 |
| 4 | `outputs_hcp/.../fold-02/.../hemi_R/template` | `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-02/group_atlas/routea_fold02_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/template` -> `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-02/group_atlas/routea_fold02_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_R/template/workbench_assets` | 2 |
| 5 | `outputs_hcp/.../fold-03/.../hemi_L/template` | `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-03/group_atlas/routea_fold03_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template` -> `outputs_hcp/reruns/shifted_cosine_crossfit_20260711/fold_group_overviews/fold-03/group_atlas/routea_fold03_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/template/workbench_assets` | 2 |

# Structural-Only Formal Output

当前三位被试 `100610 / 102311 / 102816` 的正式口径已经收敛为：

1. 皮层输入默认采用 archive 中的 `CIFTI/dtseries`
2. 海马正式结果只保留 `HippUnfold` 结构 subfield label
3. 正式图片只保留 structural 图

## 正式保留内容

- 本地输入
  - `data/hippunfold_input/sub-*/func/sub-*_task-rest_run-concat.dtseries.nii`
- 结构 label
  - `outputs/dense_corobl_batch/sub-*/hippunfold/sub-*/surf/sub-*_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
  - `outputs/dense_corobl_batch/sub-*/hippunfold/sub-*/surf/sub-*_hemi-R_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii`
- 正式 structural 图
  - `outputs/dense_corobl_batch/final_structural_only/final/sub-*/sub-*_structural.png`

## 不再视为正式结果

以下旧 volume 路线功能产物不再作为正式结论：

- `post_dense_corobl/` 下的 `reference / surface / gradients / WTA`
- `final_wb_locked/sub-*/sub-*_gradient.png`
- `final_wb_locked/sub-*/sub-*_wta.png`

这些旧产物统一移动到：

- `outputs/dense_corobl_batch/_archived_volume_functional/sub-*/`

## 正式入口

使用下列脚本完成归档旧功能结果并重建 structural-only 正式输出：

- `scripts/formalize_structural_only.py`

该入口会：

1. 验证三位被试本地 `dtseries` 已存在
2. 验证现有 `HippUnfold` 结构 label 已存在
3. 归档旧的 volume 功能结果
4. 只重建 structural 正式图

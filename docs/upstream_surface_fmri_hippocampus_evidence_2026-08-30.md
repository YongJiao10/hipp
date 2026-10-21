# HCP7T、Stanford、JJTMS 上游 surface fMRI 海马覆盖证据

调查日期：2026-08-30。本文只检查当前仓库中的非归档、非压缩内容，以及三套数据在代码/manifest 中明确指向的非归档上游文件；未读取 `archive/`、`_archive/` 或 tar/zip 等压缩归档内容。未修改或删除调查开始前已有文件；开始调查时工作区已有大量 dirty 变更，这些变更均保留。

## 判定口径

这里把“上游 hippocampus-specific surface fMRI”定义为：在进入本仓库下游处理前，已经以海马表面网格为索引的 fMRI metric/CIFTI surface 数据。CIFTI 中名为 `CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT/RIGHT` 的 grayordinates 单独列出，但它们是体素/皮层下 grayordinates，不等于海马表面顶点。

因此分开记录：

1. 上游 CIFTI/volume 是否包含海马相关信号；
2. 实际下游消费的 HippUnfold 海马表面及其 vertex 数；
3. 海马 surface fMRI 是上游已有，还是由 `volume → surface` 重采样生成。

## 总结结论

| 数据集 | 上游 CIFTI/volume 证据 | 代表性 CIFTI 结构/数量 | 实际消费的海马 surface | 上游是否已有 hippocampus-specific surface fMRI？ |
|---|---|---|---|---|
| HCP7T | `data/hippunfold_input/sub-100610/func/sub-100610_task-rest_run-concat.dtseries.nii`；另有完整 BOLD volume contract | shape `3600 × 91282`；CORTEX L/R = `29696/29716` 个 surface grayordinates，完整 fsLR 顶点各 `32492`；HIPPOCAMPUS L/R = `764/795` 个 subcortical grayordinates，无海马 surface vertex 数 | HippUnfold `space-corobl`, `den-512`, L/R 各 `512`；现存 QC raw summary 的代表 subject 为 `sub-102816` | **否**。CIFTI 有海马 subcortical grayordinates；实际海马 fMRI metric 由完整 volume 通过 Workbench 映射生成 |
| Stanford | 上游 rest CIFTI：`.../DataOrg_PFM_36P/sub-2012/func/sub-2012_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`；另有被采样流程消费的 MNI 2 mm denoised BOLD volume | shape `234 × 91282`；结构计数与上表相同（HIPPOCAMPUS L/R `764/795` 是 subcortical grayordinates） | `outputs_stanford/inputs/hippunfold/dense_corobl_batch/sub-2012/..._space-corobl_den-512_label-hipp_midthickness.surf.gii`，L/R 各 `512`；raw metric 的 summary 也记录 L/R 各 `512` | **否**。rest CIFTI 是 fsLR 皮层/皮层下 grayordinate 输入；现存海马 metric 由 volume→HippUnfold surface 生成 |
| JJTMS（旧 36P canonical） | run-level CIFTI `xcpd_surf/sub-1001/func/...run-01...space-fsLR_den-91k...dtseries.nii`，concat CIFTI 及匹配 MNI 2 mm denoised BOLD volumes 均被 subject manifest 记录 | run-01 shape `230 × 91282`；concat `686 × 91282`；CORTEX L/R = `29696/29716` surface grayordinates，完整 fsLR 顶点各 `32492`；HIPPOCAMPUS L/R `764/795` 为 subcortical grayordinates | `outputs_iowa_jj_tms/inputs/hippunfold/dense_corobl_batch/sub-1001/..._space-corobl_den-512_label-hipp_midthickness.surf.gii`，L/R 各 `512`；run-level sampled GIFTI 由 Workbench 产生 | **否**。海马 fMRI 由匹配 volume 映射到 HippUnfold surface |
| JJTMS（22P 专用） | run-level CIFTI 在 `xcpd_surf_22P/sub-1001/func/`；`run_bold_glob=null`，没有被允许用于海马采样的完整 22P denoised volume | run-01 shape `230 × 91282`；三 run concat 后 `686` timepoints；CIFTI 仍有 HIPPOCAMPUS L/R `764/795` 体素结构 | HippUnfold 结构表面本身存在（L/R 各 `512`），但 `shared_inputs_22p/routea/sub-1001` 现存输出只有 cortex reference，无 hippocampal raw metric | **不能确认有有效的 22P hippocampus-specific surface fMRI**；README 明确禁止从 CIFTI subcortical grayordinates 重建稀疏 volume |

## CIFTI header 直接检查

下列结果是 2026-08-30 用仓库约定的 `/home/yojiao/.conda/envs/py314/bin/python` 和 `nibabel` 读取二进制头得到的，不是从文件名推断：

- HCP7T：`data/hippunfold_input/sub-100610/func/sub-100610_task-rest_run-concat.dtseries.nii`，shape `(3600, 91282)`。
- Stanford：`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P/sub-2012/func/sub-2012_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`，shape `(234, 91282)`。
- JJTMS 36P：`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf/sub-1001/func/sub-1001_task-rest_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`，shape `(230, 91282)`。
- JJTMS 22P：`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf_22P/sub-1001/func/sub-1001_task-rest_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`，shape `(230, 91282)`。

四份 header 的 BrainModelAxis 均给出：

| BrainModelAxis structure | grayordinates | surface vertex 解释 |
|---|---:|---|
| `CIFTI_STRUCTURE_CORTEX_LEFT` | 29696 | fsLR full mesh `nvertices=32492`，仅其中 29696 个灰度坐标进入 CIFTI |
| `CIFTI_STRUCTURE_CORTEX_RIGHT` | 29716 | fsLR full mesh `nvertices=32492`，仅其中 29716 个灰度坐标进入 CIFTI |
| `CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT` | 764 | subcortical grayordinates；没有 surface mesh vertex 数 |
| `CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT` | 795 | subcortical grayordinates；没有 surface mesh vertex 数 |

其余 grayordinates 是脑干、小脑、丘脑、纹状体、杏仁核等 subcortical/小脑结构；它们不构成 HippUnfold 海马表面 fMRI。

## HCP7T 证据

### 上游路径与 CIFTI 使用

- `scripts/core_workflow/README.md:28` 把 HCP HippUnfold structural 输入根定义为 `data/hippunfold_input`，输出为 `outputs_hcp/inputs/hippunfold/dense_corobl_batch/`；`:32-35` 固定 `density=512`、`space=corobl`。
- `scripts/core_workflow/02_01_prepare_group_inputs.py:18` 定义 HCP 默认完整 BOLD volume root 为 `/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume`；`:25-28` 明确 physical input 可以是完整 denoised run-level volume，dtseries 只定义 run roster/timepoint pairing。
- `scripts/core_workflow/03_02_build_kong2022_routea_reference.py:23-33` 将 CIFTI 定义为 cortical network reference 的输入；`:216-220` 通过 `extract_structure_data` 只提取 `CORTEX_LEFT`/`CORTEX_RIGHT`。
- `scripts/core_workflow/lib/extract_cortex_merged_network_timeseries.py:8-18` 要求被提取结构暴露 `vertex`，因此该提取器的目标是表面皮层结构；它没有把 `HIPPOCAMPUS_*` 转成 HippUnfold 海马表面。

### 海马 surface 是如何出现的

- `scripts/core_workflow/lib/surface_sampling.py:52-70` 为 L/R 查找 HippUnfold `midthickness.surf.gii`，然后运行 Workbench `-volume-to-surface-mapping`，输入参数是 BOLD volume 和海马 midthickness surface。
- `scripts/core_workflow/lib/surface_sampling.py:85-95` 的 summary 字段包括 `surface`、`metric_gii`、`n_vertices`、`n_timepoints`、`space` 和 mapping method。
- `scripts/core_workflow/lib/surface_sampling.py:106-121` 明确 run-level Workbench 每次接收一个 source volume；临时 GIFTI 返回时移除，不写入 concat volume/CIFTI。
- `scripts/core_workflow/02_01_prepare_group_inputs.py:216-240` 将运行参数固定为 `density="512"`、`space="corobl"`、`mapping_method="trilinear"`，并将 `surface_sampling_source` 写为 `"volume"`；`:283-292` 进一步说明 volume 被采样到 HippUnfold geometry，CIFTI subcortical grayordinates 不能替代完整 volume。

### 现存输出元数据

`outputs_hcp/inputs/qc/vertex_qc_t1only/shared_inputs_166/sub-102816/surface/raw/surface_sampling_summary.json:1-20` 记录：

- L/R consumed geometry 是 `...sub-102816..._space-corobl_den-512_label-hipp_midthickness.surf.gii`（`:4`, `:14`）；
- L/R raw metric 是 `...sub-102816..._space-corobl_den-512_label-hipp_bold.func.gii`（`:5`, `:15`）；
- `n_vertices=512`、`n_timepoints=3600`、`space=corobl`、`mapping_method=trilinear`（`:6-10`, `:16-20`）。

所以 HCP7T 的“海马 surface fMRI”确实存在于现存下游输入，但 provenance 是完整 BOLD volume → HippUnfold surface 的重采样，而不是上游 fsLR CIFTI 自带的 hippocampal cortical surface。

## Stanford 证据

### 上游 CIFTI 与 volume

- `scripts/core_workflow/README.md:29` 给出 Stanford structural input root 与 `outputs_stanford/inputs/hippunfold/dense_corobl_batch/`；上游 functional CIFTI 和 volume 的实际绝对路径由下方 output manifests 记录。
- `outputs_stanford/inputs/kong2022/shared_inputs/surface/sub-2012/surface/raw/stage_manifest.json:2-12` 的 command 是 `lib.surface_sampling.sample_hipp_surface_timeseries`，实际输入是 `.../sub-2012/func/sub-2012_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz`。
- `outputs_stanford/inputs/kong2022/shared_inputs/routea/sub-2012/reference/kong2022/reference_summary.json:6` 记录被 reference stage 消费的 fsLR 91k CIFTI；`:46` 记录其 `234` timepoints，`:54-66` 记录左右 cortex grayordinates。

### 现存 consumed surface 与生成 metric

`outputs_stanford/inputs/kong2022/shared_inputs/surface/sub-2012/surface/raw/surface_sampling_summary.json:1-20` 记录：

- L/R consumed HippUnfold midthickness 分别在 `outputs_stanford/inputs/hippunfold/dense_corobl_batch/sub-2012/..._hemi-L/R_space-corobl_den-512_label-hipp_midthickness.surf.gii`（`:4`, `:14`）；
- 生成的 raw GIFTI 为 `..._hemi-L/R_space-corobl_den-512_label-hipp_bold.func.gii`（`:5`, `:15`）；
- L/R 均为 `n_vertices=512`、`n_timepoints=234`、`mapping_method=trilinear`（`:6-10`, `:16-20`）。

因此 Stanford 的 rest CIFTI 虽然包含 `HIPPOCAMPUS_LEFT/RIGHT` grayordinates，但当前海马 surface fMRI 不是从这些 grayordinates 直接读出，而是从完整 MNI BOLD volume 生成。

`outputs_stanford/inputs/kong2022/prepare_t1only_group_inputs_manifest.json:6-13` 记录 Stanford 的 HippUnfold/shared roots 和 194/194/194 的完成计数；其中 `shared_inputs/surface_151` 与实际 enumerated `shared_inputs/surface` 存在路径迁移/别名差异，不能据此把 CIFTI 误认为海马 surface。

## JJTMS 证据

### 上游 CIFTI、volume 与结构选择

- 对旧 36P canonical，`scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py:79-93` 固定 volume root `xcpd/restfmri`、run-level CIFTI root `xcpd_surf`、concat CIFTI root `xcpd_surf_concat` 和 HippUnfold root。
- 该 matcher 的 `:216-269` 同时解析每个 run 的 MNI 2 mm volume、fsLR 91k CIFTI 和 concat CIFTI；`:272-348` 检查 volume/CIFTI timepoints、grayordinate grid，并只记录左右 `CORTEX` vertex counts。
- 旧 36P 证据对应下方 subject manifest；其海马 surface sampling 使用完整 volume。其 CIFTI 中的海马 grayordinates 没有被当成海马 surface 输入。
- 对 22P 专用分支，原 2026-08-30 记录中完整 denoised 22P volume 尚未到位；**[2026-09-12 状态更新]**：完整的 22P 去噪 BOLD volume 已生成并挂载于 `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_vol_22P/restfmri`，覆盖全部 94 位受试者及其匹配 run。
- 依据仓库规则，严禁把 generic `xcpd/restfmri` 36P derivative、raw fMRIPrep volume 或 CIFTI subcortical grayordinates 作为 22P 海马 volume 替代；若进行 22P 海马采样，必须使用上述合法的 `xcpd_vol_22P/restfmri`。
- 因而 22P 的 `surface_functional_source="cifti-subcortex"` 只是现存 manifest 的标签，不足以证明已生成 hippocampus-specific surface fMRI；它必须服从 README 的 fail-closed 约束。
- 22P reference manifest `outputs_iowa_jj_tms/inputs/kong2022/shared_inputs_22p/routea/sub-1001/reference/kong2022/reference_summary.json:7-27` 的实际输入只有三个 run-level CIFTI；该 subject 目录枚举到的输出只有 cortex reference/common 文件，没有 surface/raw metric。
- 22P 两份 prepare manifest 的 run_bold_glob=null、CIFTI root 和不同 QC 计数（kong2022 lines 1-18；kong2022_22p lines 1-20）应视为输出元数据差异，不能推翻“缺少合法完整 22P volume”的代码/README 约束。

### JJTMS 36P canonical manifest 的实际路径与数字

`outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/sub-1001/subject_manifest.json` 提供旧 36P canonical 证据：

- `:110-139` 的 `header_evidence` 为 concat `686` timepoints、`91282` grayordinates，run lengths 为 `230/228/228`；
- `:183-221` 记录 concat CIFTI、三个 run-level fsLR 91k CIFTI，以及三个匹配的 MNI 2 mm denoised BOLD volumes；
- `:223-231` 记录 L/R consumed HippUnfold `space-corobl_den-512_label-hipp_midthickness.surf.gii`；
- `:98-108` 的 density evidence 对 L/R 都记录 `evidence_source=consumed_surface_path_and_gifti_point_count`、`n_vertices=512`。

### 海马 surface fMRI 的生成位置

- `scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py:1322-1348` 对每个 run、每个 hemi 调用 Workbench `-volume-to-surface-mapping`，输入是 `run.volume` 和 `surfaces[hemi]["consumed_asset"]`。
- `:1350-1375` 要求 sampled GIFTI shape 为 `(512, expected_timepoints)`，之后才在内存中按时间轴拼接并写出 raw hippocampal GIFTI。
- `scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py:331-348` 的 `cortex_vertex_counts` 只记录 `CORTEX_LEFT`/`CORTEX_RIGHT`；海马 CIFTI grayordinates没有进入该 cortex ROI 分支。

因此 JJTMS 旧 36P 与 HCP7T/Stanford 一样：上游 fsLR CIFTI 有海马皮层下 grayordinates，但不存在可证实为上游输入的 hippocampus-specific surface fMRI；现存 L/R `512`-vertex 海马 fMRI 是由匹配的完整体数据通过 Workbench 生成的下游 surface metric。

对于 JJTMS 22P，原 2026-08-30 记录中完整 22P volume 尚未到位；**[2026-09-12 状态更新]**：完整的 22P 去噪 BOLD volume 已生成就位（`xcpd_vol_22P/restfmri`，94 位受试者），为后续直接从 22P volume 进行海马表面采样映射提供了合法的数据源。

## 最终回答

若问题是“上游 surface fMRI 本身是否已经包含海马部分皮层”，HCP7T、Stanford 和 JJTMS 旧 36P 的答案都是：**没有 hippocampus-specific cortical surface component**。这些上游 91k CIFTI 都包含：

- 左右大脑皮层 surface grayordinates（29,696/29,716，映射到 32,492-vertex fsLR mesh）；
- 左右海马的 subcortical grayordinates（764/795），但它们不是海马表面顶点。

下游方面，HCP7T、Stanford 和 JJTMS 旧 36P 有 HippUnfold den-512 的 L/R 各 512-vertex surface metric，均来自完整 BOLD volume → 海马表面的重采样/映射，而不是上游 CIFTI 直接提供；JJTMS 22P 随着 `xcpd_vol_22P/restfmri` 的补充就位，现已具备与 36P 同等完整的结构、皮层与海马体数据输入条件。

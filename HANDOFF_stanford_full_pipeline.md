# Stanford 全流程 Handoff — 2026-06-08 05:50 UTC

## 当前状态

HippUnfold 批量运行中（Phase F），**61/199 已完成**，其余在执行中。

## 已完成

### Phase A: 代码修改（已提交？未提交）
- 新增 `scripts/core_workflow/lib/path_templates.py` — 路径模板 helper
- 新增 `tests/core_workflow/test_path_templates.py` — 9 tests (all pass)
- 修改 6 个脚本添加 `--dtseries-template` / `--bold-template` CLI:
  - `02_01_prepare_group_inputs.py`
  - `03_01_build_kong2022_reference.py`
  - `03_02_build_kong2022_routea_reference.py`
  - `04_00_materialize_cortex_regressed_surface_timeseries.py`
  - `04_01_materialize_kong2022_roi_fc.py`
- 修改 `run_hippunfold_local.sh` 支持 `HIPPUNFOLD_CORES` 环境变量
- 更新 `scripts/core_workflow/README.md`
- 修复 `02_01` 缺少的 `import subprocess`

### Phase B-D: Preflight ✅
- HippUnfold 2.0.0 环境验证通过
- 199 subjects 源文件验证通过
- RouteA 标签完美匹配（`ArealMSHBM_Kong2022/` 下 `sub-XXXX/lh_labels.txt` + `rh_labels.txt`，32492 行/文件）
- Kong2022 dlabel: `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

### Phase E: Smoke test (sub-2001) ✅
- 全流程通过：HippUnfold → surface sampling → tSNR → QC → smooth → group avg → RouteB/RouteA ref → ROI FC → intrinsic FC
- `.nii.gz` BOLD 直接可用
- `--dtseries-template` 正确读取 Stanford 格式

### Phase F: 全量运行 🔄
- **HippUnfold 批量**: 4 concurrent × **8 cores** each, xargs -P4
- 已保留 61 个完成的 subject
- 批次启动时间: 2026-06-08 00:39 UTC

## 环境变量（所有命令使用）
```bash
PY314=/home/yojiao/.conda/envs/py314/bin/python
STANFORD_ROOT=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P
AREA_ROOT=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022
STANFORD_OUT=outputs_stanford
HIPPO_BATCH=${STANFORD_OUT}/inputs/stanford/hippunfold/dense_corobl_batch
KONG_ROOT=${STANFORD_OUT}/inputs/stanford/kong2022
SURFACE_ROOT=${KONG_ROOT}/shared_inputs/surface_151
ROUTEA_ROOT=${KONG_ROOT}/shared_inputs/routea_151
ROUTEB_ROOT=${KONG_ROOT}/shared_inputs/routeb_151
TEMPLATE_ROOT=${KONG_ROOT}/templates/hipp_group_average_t1only_151
ROUTEA_LABEL_ROOT=${AREA_ROOT}    # 直接指向源数据，无需桥接
MANIFEST=${KONG_ROOT}/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt
KONG2022_DLABEL=/home/yojiao/hpchome/hipp/external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii

# 路径模板（Stanford 实际命名）
DT_TEMPLATE='{data_root}/{sub_subject}/func/{sub_subject}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
BOLD_TEMPLATE='{bold_root}/{sub_subject}/func/{sub_subject}_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'
LEFT_SURF_TEMPLATE='{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_midthickness.surf.gii'
RIGHT_SURF_TEMPLATE='{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_midthickness.surf.gii'
```

## 检查 HippUnfold 进度
```bash
find outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch -maxdepth 5 -name '*_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii' -path '*/surf/*' 2>/dev/null | wc -l
# 期望: 199/199
```

## HippUnfold 完成后：执行下游步骤

### 先决条件
- HippUnfold 199/199 完成
- xargs 进程已退出
- 无残留 snakemake 进程

### 步骤

**F3: 检查输出 (01_02)**
```bash
"${PY314}" scripts/core_workflow/01_02_check_hippunfold_outputs.py \
  --dataset stanford \
  --input-root "${STANFORD_ROOT}"
```

**F4: 群组输入准备 (02_01)**
```bash
"${PY314}" scripts/core_workflow/02_01_prepare_group_inputs.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --route-root "${KONG_ROOT}" \
  --shared-root "${SURFACE_ROOT}" \
  --template-root "${TEMPLATE_ROOT}" \
  --bold-root "${STANFORD_ROOT}" \
  --bold-template "${BOLD_TEMPLATE}" \
  --subjects-manifest "${KONG_ROOT}/manifests/stanford_all_subjects.txt" \
  --jobs 2
```

**F5: QC Manifest (02_02)**
```bash
"${PY314}" scripts/core_workflow/02_02_build_qc_manifest.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --shared-root "${SURFACE_ROOT}" \
  --out-dir "${KONG_ROOT}/qc" \
  --manifest-dir "${KONG_ROOT}/manifests"
```

**F6: RouteB Reference (03_01)**
```bash
"${PY314}" scripts/core_workflow/03_01_build_kong2022_reference.py \
  --shared-root "${ROUTEB_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 4
```

**F7: RouteA Reference (03_02)**
```bash
"${PY314}" scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  --shared-root "${ROUTEA_ROOT}" \
  --route-a-root "${ROUTEA_LABEL_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 4
```

**F8: RouteB FC (04_01)**
```bash
"${PY314}" scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --reference-root "${ROUTEB_ROOT}" \
  --mask-root "${SURFACE_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --left-cortex-surface-template "${LEFT_SURF_TEMPLATE}" \
  --right-cortex-surface-template "${RIGHT_SURF_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

**F10: Intrinsic FC (04_02)**
```bash
"${PY314}" scripts/core_workflow/04_02_materialize_intrinsic_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

**F11: Gradients (05_01)** — jobs=1 (计算密集)
```bash
"${PY314}" scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

**F12: Group Atlas (05_02)** — jobs=1
```bash
"${PY314}" scripts/core_workflow/05_02_run_group_atlas.py \
  --dataset stanford \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

**F13: Visualization (06_01, 07_01)**
```bash
"${PY314}" scripts/core_workflow/06_01_build_group_overview.py --dataset stanford
"${PY314}" scripts/core_workflow/07_01_build_group_matrix.py --dataset stanford
```

## 重启 HippUnfold 批量（如需）

```bash
# 单 subject worker: outputs_stanford/work/run_one_hippunfold.sh <SID>
# 批量启动 (4 concurrent × 8 cores):
SCRIPT=/home/yojiao/hpchome/hipp/outputs_stanford/work/run_one_hippunfold.sh
MANIFEST=outputs_stanford/inputs/stanford/kong2022/manifests/stanford_all_subjects.txt
sed 's/^sub-//' "${MANIFEST}" | xargs -P 4 -I {} "${SCRIPT}" {} &
```

## 关键文件路径

| 用途 | 路径 |
|------|------|
| 执行计划 | `stanford_full_pipeline_plan.md` |
| HippUnfold worker | `outputs_stanford/work/run_one_hippunfold.sh` |
| Path templates | `scripts/core_workflow/lib/path_templates.py` |
| 测试 | `tests/core_workflow/test_path_templates.py` |
| 修改后的 shell | `scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh` |

## 注意
- **Python**: 必须用 `/home/yojiao/.conda/envs/py314/bin/python`
- **HippUnfold**: 通过脚本自动激活 `hippo2` conda env
- **并发**: 全流程 ≤4 (HippUnfold), Workbench 步骤 ≤2, 梯度/atlas=1
- **dice threshold**: smoke 用 0.60, 全量用默认 0.70
- **HippUnfold cores**: 环境变量 `HIPPUNFOLD_CORES=8` 已设

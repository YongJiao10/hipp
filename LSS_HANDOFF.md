# Handoff: LSS esTT — HippUnfold 重跑 + Cortex PFM + Cluster FC

**更新时间：** 2026-04-24 ~02:34 UTC  
**当前状态：** ✅ HippUnfold 25/25 完成（36P）；✅ 24P Step B/C 完成；⏳ 24P Step D 运行中（PID 1667288）

---

## 路径约定

| 变量 | 路径 |
|------|------|
| LSS 根 | `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/` |
| LSS 工作目录 | `{LSS根}/DataOrg_PFM_36P/hipp_outputs/` |
| 本地脚本 | `{LSS根}/src/hipp/` |
| hipp repo | `/home/yojiao/hpchome/hipp/`（脚本/资产来源，不写入结果） |

---

## 当前运行进程

| PID | 角色 | 状态 | 日志 |
|-----|------|------|------|
| 1049857 | 原始批次管理进程（24/25完成后退出） | ✅ 已完成 | `DataOrg_PFM_36P/hipp_outputs/logs/hippunfold_batch_t1only.log` |
| 380641 | sub-846 第3次手动重启 | ✅ 已完成（363/363 steps） | `DataOrg_PFM_36P/hipp_outputs/logs/hippunfold_846_restart3.log` |
| — | 24P Step B 海马 surface 采样 | ✅ 已完成（25/25） | `DataOrg_PFM_24P/hipp_outputs/logs/sampling_24p_batch.log` |
| — | 24P Step C Cortex PFM + 网络时序 | ✅ 已完成（25/25） | `DataOrg_PFM_24P/hipp_outputs/logs/cortex_pfm_24p.log` |
| 1667288 | 24P Step D Cluster FC | 🔄 运行中（25×4 combos=100个任务） | `DataOrg_PFM_24P/hipp_outputs/logs/cluster_fc_24p.log` |

```bash
# 完成数
ls /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/dense_corobl_batch_t1only/sub-*/hippunfold/sub-*/surf/*hemi-L*space-corobl*den-512*label-hipp_midthickness.surf.gii 2>/dev/null | wc -l
# 期望最终: 25

# sub-846 进度
ps -p 380641
tail -20 /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/logs/hippunfold_846_restart3.log
```

### sub-846 历史与当前状态

sub-846 是最后一个（第25个）subject，经历了3次重启：

| 次数 | 失败原因 | 处理 |
|------|----------|------|
| 原始批次 | `template_shape_reg` greedy 卡住4.5h，随后 `laynii_layers_equidist` subprocess 挂死 | kill PID 404950 |
| 第1次重启 | 用错脚本（`run_hippunfold_local.sh` 而非 isolated_cache），asset store 未命中，重新下载 | kill PID 723484 |
| 第2次重启 | hemi-R `nnunet_dsegsplit` 已存在但 `nnunet_dseg.nii.gz`（temp）被清理，snakemake 无法重建 | 删除 dsegsplit，第3次重启 |
| **第3次重启** | **使用正确脚本+asset store，从 nnUNet hemi-R 重跑** | **PID 380641，进行中** |

### 如果 sub-846 再次失败需重启

```bash
LSS_ROOT="/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT"
LSS_HIPP_OUT="$LSS_ROOT/DataOrg_PFM_36P/hipp_outputs"

# 1. unlock
HIPPUNFOLD_EXTRA_ARGS="--unlock" \
HIPPUNFOLD_ASSET_STORE_ROOT="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared" \
  bash /home/yojiao/hpchome/hipp/scripts/run_hippunfold_local_t1w_isolated_cache.sh \
    846 "$LSS_HIPP_OUT/staging_input" \
    "$LSS_HIPP_OUT/dense_corobl_batch_t1only/sub-846/hippunfold"

# 2. restart
nohup env \
  HIPPUNFOLD_EXTRA_ARGS="--rerun-incomplete" \
  HIPPUNFOLD_ASSET_STORE_ROOT="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared" \
  bash /home/yojiao/hpchome/hipp/scripts/run_hippunfold_local_t1w_isolated_cache.sh \
    846 "$LSS_HIPP_OUT/staging_input" \
    "$LSS_HIPP_OUT/dense_corobl_batch_t1only/sub-846/hippunfold" \
  >> "$LSS_HIPP_OUT/logs/hippunfold_846_restartN.log" 2>&1 &
echo "PID: $!"
```

> ⚠️ **关键**：必须用 `run_hippunfold_local_t1w_isolated_cache.sh`（不是 `run_hippunfold_local.sh`），且必须设置 `HIPPUNFOLD_ASSET_STORE_ROOT`，否则会从 Zenodo 重新下载模板。

### 如果整批需重启（25人）

```bash
cd /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT
nohup /home/yojiao/.conda/envs/py314/bin/python src/hipp/run_lss_hippunfold_batch.py \
  --concurrency 1 \
  >> DataOrg_PFM_36P/hipp_outputs/logs/hippunfold_batch_t1only.log 2>&1 &
echo "PID: $!"
```

---

## HippUnfold 重跑历史与当前方案

### 版本历史

| 目录 | modality | t1-reg-template | 状态 |
|------|----------|-----------------|------|
| `dense_corobl_batch_legacy/` | T2w | 无 | 旧结果，保留 |
| `dense_corobl_batch_t2t1_incomplete/` | T2w | 有 | 中途停止（5/25完成），保留 |
| `dense_corobl_batch_t1only/` | **T1w** | 无 | **当前运行** |

### 为什么改成 T1w only

T2w+t1-reg-template 方案中途停止，改用 T1w only 重新开始。Task101（T1w）nnUNet 模型已下载（2.3G tar，在 asset store 中），首次运行时 snakemake 会自动解压。

### 关键参数

| 参数 | 值 |
|------|-----|
| modality | `T1w` |
| t1-reg-template | 无 |
| output-density | `512` |
| output-spaces | `corobl` |
| input_dir | `hipp_outputs/staging_input/`（仅含 T1w/T2w 的干净目录） |
| output_dir | `hipp_outputs/dense_corobl_batch_t1only/sub-{ID}/hippunfold/` |
| cache base | `hipp_outputs/runtime/hippunfold_lss_t1only/` |
| nnUNet 模型 | `hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar` |
| concurrency | 1（串行，约 44 分钟/人） |
| shell 脚本 | `hipp/scripts/run_hippunfold_local_t1w_isolated_cache.sh` |

### 路径

```
hipp_outputs/
  dense_corobl_batch_legacy/          ← 旧结果（T2w only, no t1-reg-template）
  dense_corobl_batch_t2t1_incomplete/ ← 中途停止（T2w + t1-reg-template，5/25）
  dense_corobl_batch_t1only/          ← 当前结果（正在写入）
  staging_input/sub-{ID}/anat/        ← T1w + T2w 入口（symlink）
  runtime/hippunfold_lss_t1only/      ← 每人临时 workdir（运行完自动清理）
  logs/hippunfold_{ID}.log            ← 每人详细日志
  logs/hippunfold_batch_t1only.log    ← 批处理汇总日志
```

---

## HippUnfold 完成后的下游任务

### Step A：QC（重新评估）

旧 QC 结论（基于 legacy 结果）：23/25 pass（sub-677 Dice-L=0.234，sub-778 Dice-R=0.583 失败）。  
新跑完成后**必须重新跑 QC**，标准与 HCP 项目相同：

1. HippUnfold Dice ≥ 0.70（两侧）— 看 `sub-{ID}_desc-unetf3d_dice.tsv`
2. fMRI null vertex % < 20%（sd == 0，PSC 单位）

QC 报告位置：`hipp_outputs/qc/hippunfold_qc_lss_esTT_25subjects.md`（需更新）

### Step B：海马 surface 功能时序采样

旧采样结果在 `hipp_outputs/hipp_functional_parcellation_network/shared/sub-{ID}/surface/raw/`，基于 legacy HippUnfold surface。新 HippUnfold 完成后需重新采样。

参照 `run_one_subject.py` / `run_all_subjects_serial.sh`。

### Step C：Cortex PFM + 网络时序提取（已完成，无需重跑）

25 人已完成，结果在：
- `cortex_pfm/sub-{ID}/kong2019/pfm_summary.json` ✅
- `shared/sub-{ID}/reference/cortex_canonical_network_timeseries.npy`（shape: 8×593）✅
- 使用 `extract_cortex_merged_network_timeseries.py`（直接 vertex 级别 merge，非 ROI 中间步骤）
- 皮层 valid-vertex 掩码：sd > 0（PSC 单位）

### Step D：Cluster FC 矩阵（需在 Step B 完成后重跑）

脚本：`src/hipp/compute_lss_cluster_fc.py`  
当前状态：23 人 QC-pass 集已跑完，但依赖旧 surface → **Step B 重采样后需重跑**

```bash
cd /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT
/home/yojiao/.conda/envs/py314/bin/python src/hipp/compute_lss_cluster_fc.py \
  --manifest DataOrg_PFM_36P/hipp_outputs/manifests/lss_esTT_23subjects_qcpass.txt
```

4 个 atlas combo × 23 人 × 2 hemi = 184 个输出文件  
输出：`hipp_outputs/cluster_fc/{combo}/sub-{ID}/hemi-{L,R}_cluster_network_fc.npy`

---

## HippUnfold 环境信息

| 项目 | 值 |
|------|-----|
| 环境 | `hippo2`（hippo2 repo clone at `hipp/.cache/conda_envs/hippo2/`） |
| hippunfold 版本 | 2.0.0（CLI 实测）|
| nnUNet T1w 模型 | `hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar`（2.3G，已下载） |
| nnUNet T2w 模型 | `hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task102_hcp1200_T2w.nnUNetTrainerV2.model_best`（已解压） |
| greedy/c3d/c3d_affine_tool | 均在 hippo2 conda env，无需外部 binary dir |
| Python env（下游） | `/home/yojiao/.conda/envs/py314/bin/python` |

---

---

## 24P 下游分析规划

### 背景

`DataOrg_PFM_24P/` 目录已存在，包含 25 人的 24P 去噪 bold/dtseries，anat 与 36P 完全相同。HippUnfold 输出（anatomy-based）**不需要重跑**，直接复用 36P 的 `dense_corobl_batch_t1only/`。

### 资源复用 vs 重新生成

| 资源 | 来源 |
|------|------|
| HippUnfold surfaces | 复用 `DataOrg_PFM_36P/hipp_outputs/dense_corobl_batch_t1only/` |
| Group atlases | 复用 `DataOrg_PFM_36P/hipp_outputs/group_atlases/` |
| HippUnfold Dice QC | 复用 36P 结论（anatomy 相同）|
| fMRI null vertex QC | **需重检**（24P 去噪不同，sd==0 vertex 可能不同）|
| 海马 surface 时序 | **重新采样**（用 24P bold）→ `DataOrg_PFM_24P/hipp_outputs/...` |
| Cortex PFM + 网络时序 | **重新跑**（用 24P dtseries）→ `DataOrg_PFM_24P/hipp_outputs/...` |
| Cluster FC 矩阵 | **重新跑**（用 24P 采样结果）→ `DataOrg_PFM_24P/hipp_outputs/...` |

### Step B：24P 海马 surface 采样

直接调采样脚本（跳过 `run_one_subject.py`，避免重跑 HippUnfold）：

```bash
PYTHON=/home/yojiao/.conda/envs/py314/bin/python
SRC=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/src/hipp
ROOT36=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P
ROOT24=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_24P
HIPP_OUT24="$ROOT24/hipp_outputs"

for SID in $(cat "$ROOT36/hipp_outputs/manifests/lss_esTT_25subjects.txt"); do
  $PYTHON "$SRC/common/sample_hipp_surface_timeseries.py" \
    --bold "$ROOT24/sub-${SID}/func/sub-${SID}_task-rest_run-concat_desc-denoised_bold.nii.gz" \
    --hippunfold-dir "$ROOT36/hipp_outputs/dense_corobl_batch_t1only/sub-${SID}/hippunfold" \
    --subject "$SID" --density 512 --space corobl \
    --mapping-method trilinear --smooth-iters 0 \
    --smooth-fwhm-mm 2.0 4.0 \
    --outdir "$HIPP_OUT24/hipp_functional_parcellation_network/shared/sub-${SID}/surface/raw" \
    >> "$HIPP_OUT24/logs/sampling_24p_${SID}.log" 2>&1 \
    && echo "done $SID" || echo "FAIL $SID"
done
```

### Step C：24P Cortex PFM

`run_lss_cortex_pfm_batch.py` 的 `sub_dir` 硬编码到 36P，**需要加 `--data-root` 参数**（小改动）。改动后：

```bash
$PYTHON $SRC/run_lss_cortex_pfm_batch.py \
  --data-root $ROOT24 \
  --manifest $ROOT36/hipp_outputs/manifests/lss_esTT_25subjects.txt \
  --out-root $ROOT24/hipp_outputs/cortex_pfm \
  --shared-root $ROOT24/hipp_outputs/shared \
  --concurrency 4
```

> ⚠️ `run_lss_cortex_pfm_batch.py` 需要先加 `--data-root` CLI 参数，修改 `sub_dir = Path(args.data_root) / f"sub-{subject}"`

### Step D：24P Cluster FC

脚本已完全可配置，无需修改：

```bash
$PYTHON $SRC/compute_lss_cluster_fc.py \
  --manifest $ROOT36/hipp_outputs/manifests/lss_esTT_25subjects.txt \
  --atlas-root $ROOT36/hipp_outputs/group_atlases \
  --hipp-shared $ROOT24/hipp_outputs/hipp_functional_parcellation_network/shared \
  --cortex-shared $ROOT24/hipp_outputs/shared \
  --out-root $ROOT24/hipp_outputs/cluster_fc
```

### 当前状态

- [ ] fMRI null vertex QC（24P）
- [x] `run_lss_cortex_pfm_batch.py` 加 `--data-root` 参数（已完成）
- [x] Step B：25 人海马 surface 采样（24P bold）— 25/25 ✅
- [x] Step C：Cortex PFM（24P dtseries）— 25/25 ✅
- [ ] Step D：Cluster FC（24P）— 运行中 PID 1667288，100个任务

Step D 完成后检查：
```bash
tail -5 /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_24P/hipp_outputs/logs/cluster_fc_24p.log
```

Step D 重启命令（如果进程不在）：
```bash
ROOT36="/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P"
ROOT24="/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_24P"
nohup /home/yojiao/.conda/envs/py314/bin/python \
  /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/src/hipp/compute_lss_cluster_fc.py \
  --manifest "$ROOT36/hipp_outputs/manifests/lss_esTT_25subjects.txt" \
  --atlas-root "$ROOT36/hipp_outputs/group_atlases" \
  --hipp-shared "$ROOT24/hipp_outputs/hipp_functional_parcellation_network/shared" \
  --cortex-shared "$ROOT24/hipp_outputs/shared" \
  --out-root "$ROOT24/hipp_outputs/cluster_fc" \
  >> "$ROOT24/hipp_outputs/logs/cluster_fc_24p.log" 2>&1 &
echo "PID: $!"
```

---

## 24P 启动时踩到的坑（2026-04-24）

### 坑1：`run_lss_cortex_pfm_batch.py` 漏传 `--cross-atlas-merge-json`
- **原因**：`extract_cortex_merged_network_timeseries.py` 在 Apr 22 16:41 将该参数从可选（有 default）改为 `required=True`；36P 跑于 Apr 22 14:14（当时是可选），批处理脚本沿用旧调用方式未同步。
- **修复**：`run_lss_cortex_pfm_batch.py` extract_cmd 中补加 `--cross-atlas-merge-json`，指向 `HIPP_REPO/config/cross_atlas_network_merge.json`。

### 坑2：`extract_cortex_merged_network_timeseries.py` 严格检查 Language/Limbic
- **原因**：Apr 22 版本新增了"所有 canonical_network_order 中的网络必须存在"的检查，但 `canonical_network_order` 包含10个网络（含 Language/Limbic），而 kong2019 只有8个（无 Language/Limbic）。
- **修复**：`extract_cortex_merged_network_timeseries.py` 中将 `canonical_names` 改为只取 atlas mapping 实际覆盖的网络（`set(mapping.values()) - exclude_labels`），不覆盖的网络不检查也不输出。

---

## 旧 Cortex PFM 方法论说明（已修正）

旧流程错误：通过 ROI component 中间层（等权平均）→ 系统性低估顶点数多的子网络贡献。  
新流程：raw FASTANS label.gii → 直接按 vertex 累加到 8 个 canonical 网络 → 算术平均（`extract_cortex_merged_network_timeseries.py`）。  
此修正已生效，cortex 网络时序已重新提取。

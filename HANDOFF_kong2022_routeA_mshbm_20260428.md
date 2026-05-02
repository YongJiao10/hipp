# Handoff: Kong2022 Route A (MATLAB-only on SGE)

## 1) 当前任务需求（用户意图）
- 只做 Route A 最小闭环里的 MATLAB 部分（Kong2022_ArealMSHBM 个体化分区）。
- 必须通过 `sh` 提交作业到队列跑，不能在 login node 直接算。
- 不要再跑 Kong2019 路径；目标是 Kong2022_ArealMSHBM。
- 用户环境明确要求用 `py313`（因为 Kong2022 的 MATLAB 流程内部会调用 Python diffusion embedding 脚本）。

## 2) 我做了什么

### 2.1 已完成的方向修正
- Route A 脚本已切到 Kong2022_ArealMSHBM（不是 Kong2019）。
- 输出根目录默认改到 `outputs_experiments/cortex_kong2022_areal_mshbm`。

### 2.2 新增/修改的提交脚本（用于仅 MATLAB 阶段）
- 新增：`tmp/matlab_kong2022_mshbm_subject_sge.sh`
  - SGE array task：2 个被试（100610, 102311）
  - 作业节点内做：
    - concat dtseries 拆 4 run
    - 生成 run list / censor list
    - 调 MATLAB `CBIG_ArealMSHBM_parcellation_single_subject`
  - 关键设置：
    - `export PATH=${repo_root}/software/workbench/bin_linux64:${PATH}`
    - `params.overwrite_flag=0`（避免后续失败重跑时每次全量清空重算）
- 已多次通过：
  - `sh -c 'qsub tmp/matlab_kong2022_mshbm_subject_sge.sh'`

### 2.3 已修复/补齐的依赖（本地 external/CBIG）
从 `/tmp/cbig_upstream` 同步到本仓库 `external/CBIG`：
- `stable_projects/brain_parcellation/Kong2022_ArealMSHBM`
- `utilities/matlab/FC/CBIG_ComputeCorrelationProfile.m`
- `external_packages/matlab/default_packages/mtimesx_20110223`
- `utilities/matlab/speedup_gradients`
- `utilities/matlab/fslr_matlab`
- `external_packages/SD/SDv1.5.1-svn593/BasicTools/read_surf.m`
- `data/templates/surface/fs_LR_32k`
- `external_packages/matlab/default_packages/cifti-matlab`
- `external_packages/matlab/non_default_packages/cifti-matlab-WashU-gradient`
- `external_packages/matlab/default_packages/WashU_gradients`
- `external_packages/matlab/default_packages/others/constructNPYheader.m`
- 最后又把：
  - `external_packages/matlab/default_packages/*`
  - `external_packages/matlab/non_default_packages/*`
  全量补齐（防止遗漏 helper）。

### 2.4 关键脚本修正（已按用户要求固定 py313）
- 修改：`external/CBIG/utilities/matlab/speedup_gradients/CBIG_SPGrad_diffusion_embedding.sh`
  - 现在固定：
    - `source /Users/yojiao/miniconda3/etc/profile.d/conda.sh`
    - `conda activate py313`
  - 已删除 `CBIG_py3` fallback。

## 3) 目前状态（交接时）

### 3.1 队列作业
- 最新提交作业：`5885248`（array tasks `1,2`）
- 最近一次 `qstat` 显示：2 个 task 都是 `r`
- 但异常点：
  - `/Users/yojiao/logs/kong2022_mshbm.5885248.1.out` 和 `.2.out` 大小是 `0`
  - `matlab_wrapper.log` 仍停在 `2026-04-28 03:05`（旧轮次），未见新写入
- 这意味着：当前 `5885248` 可能在 scheduler 侧显示运行，但没有有效输出流（需新 agent 核查）。

### 3.2 产物是否跑通
- 截止交接时，仍**未生成**：
  - `project_dir/route_a_outputs/lh_labels.txt`
  - `project_dir/route_a_outputs/rh_labels.txt`
- 即：MATLAB 个体化分区尚未最终跑通。

## 4) 历史主要失败原因（已定位）
按时间序：
1. `censor list and fMRI list cannot be empty!` -> 已改为显式 censor 文件。
2. 缺 `CBIG_SPGrad_RSFC_gradients` -> 已补 speedup_gradients。
3. 缺 `CBIG_read_fslr_surface` -> 已补 fslr_matlab。
4. 缺 `read_surf` -> 已补 SD BasicTools。
5. 缺 `ft_read_cifti`/`ciftiopen` -> 已补 cifti/WashU 包。
6. 缺模板 `lh.sphere` -> 已补 `data/templates/surface/fs_LR_32k`。
7. 缺 `constructNPYheader` -> 已补 `default_packages/others`。
8. `source: CBIG_init_conda: file not found` 导致 diffusion embedding 缺失 -> 已把 diffusion 脚本固定为 `py313` 激活路径。

## 5) 新 agent 下一步（建议按顺序）

1. 先确认 `5885248` 是否真在工作：
   - `qstat -u yojiao`
   - 若仍 `r` 但输出文件一直 0、且 `matlab_wrapper.log` 不更新，判定为假运行/僵死。

2. 若僵死，先清掉该轮并重提：
   - `qdel 5885248`（如仍在跑）
   - `sh -c 'qsub tmp/matlab_kong2022_mshbm_subject_sge.sh'`

3. 重新提交后，优先看是否命中新的 py313 路径：
   - 在 `matlab_wrapper.log` 里确认不再出现 `CBIG_init_conda` missing。

4. 以这两个文件作为“跑通”判据：
   - `.../sub-100610/.../project_dir/route_a_outputs/lh_labels.txt`
   - `.../sub-100610/.../project_dir/route_a_outputs/rh_labels.txt`
   - `.../sub-102311/.../project_dir/route_a_outputs/lh_labels.txt`
   - `.../sub-102311/.../project_dir/route_a_outputs/rh_labels.txt`

5. MATLAB 跑通后（仅在需要时）再做后处理：
   - dlabel/绘图可以留给用户指定的 Python 系统执行。

## 6) 这次会话内新增/修改的关键文件列表
- `tmp/matlab_kong2022_mshbm_subject_sge.sh` (新增，多次迭代)
- `external/CBIG/utilities/matlab/speedup_gradients/CBIG_SPGrad_diffusion_embedding.sh` (固定 py313)
- 以及上文 2.3 列出的 `external/CBIG` 同步目录（大量依赖补齐）

## 7) 备注
- 用户明确要求：只通过提交作业在计算节点跑，不在 login node 直接跑计算。
- 用户明确要求：环境用 `py313`，不要再模糊 fallback。

## 8) 2026-04-28 13:20 接手后的最新状态

### 已补齐/修改
- 补齐 CBIG vendored Python 依赖：
  - `external/CBIG/external_packages/python/mapalign-master`
- 补齐 CBIG fsLR 900 template：
  - `external/CBIG/data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii`
- 补齐 MATLAB helpers：
  - `external/CBIG/utilities/matlab/stats/CBIG_corr.m`
  - `external/CBIG/utilities/matlab/parcellation/*`
- 本地修改 `CBIG_ComputeConnectedComponentsFromSurface.m`：
  - 用 MATLAB `graph/conncomp` 替代旧 `matlab_bgl` `components_mex` 路径，避免依赖 MEX。
- 更新 `tmp/matlab_kong2022_mshbm_subject_sge.sh`：
  - `mem_free=32G`
  - `-tc 1`
  - 已有 `route_a_outputs/lh_labels.txt` 和 `rh_labels.txt` 时直接 skip
  - preflight 增加 `mapalign`、`CBIG_corr`、connected-components helper、fsLR 900 template。
- 已同步文档：
  - `docs/hipp_parcellation_network/kong2022_route_a_smoke.md`
  - `kong2022_route_a_mshbm_plan.md`
  - `docs/environment/hippo2_reproducibility_manifest.md`

### 已跑过的 SGE 作业
- `5885563`
  - 越过 `mapalign`，生成左右 hemisphere embedding。
  - 失败于缺 `fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii`。
- `5885629`
  - 越过 template，生成 profile 阶段。
  - 失败于缺 `CBIG_corr`。
- `5885649`
  - 两个 subject 都生成四个 session profile。
  - `sub-100610` 失败于缺 `CBIG_ComputeConnectedComponentsFromSurface`。
  - `sub-102311` 在 EM 附近被 exit `137` kill。
- `5885719`
  - 使用 `mem_free=32G`、`-tc 1` 串行。
  - `sub-100610` 和 `sub-102311` 都在日志到达：
    - `It is EM iteration.. 2..`
    - `Estep..`
  - 随后被 SGE exit `137 (Killed)`。

### 当前产物状态
- 已有：
  - 两个 subject 的 `lh/rh_emb_100_distance_matrix.mat`
  - 两个 subject 的 4 个 session profile `.mat`
- 仍未生成：
  - `outputs_experiments/cortex_kong2022_areal_mshbm/sub-100610/kong2022_areal_mshbm/project_dir/route_a_outputs/lh_labels.txt`
  - `outputs_experiments/cortex_kong2022_areal_mshbm/sub-100610/kong2022_areal_mshbm/project_dir/route_a_outputs/rh_labels.txt`
  - `outputs_experiments/cortex_kong2022_areal_mshbm/sub-102311/kong2022_areal_mshbm/project_dir/route_a_outputs/lh_labels.txt`
  - `outputs_experiments/cortex_kong2022_areal_mshbm/sub-102311/kong2022_areal_mshbm/project_dir/route_a_outputs/rh_labels.txt`

### 当前判断
- 缺依赖问题已基本清完，当前不是 preflight/CBIG path 问题。
- 剩余 blocker 是 MATLAB EM 个体化阶段在计算节点被外部 kill（SGE exit `137`），发生在第二轮 EM 的 `Estep..` 后。
- 不建议未经用户确认把算法改成少 session、少 ROI、跳过 connectedness、提前停止 EM；这会偏离 Route A 语义。

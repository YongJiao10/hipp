# HCP 3T Task-fMRI 每人必备文件清单（EMOTION + WM 优先）

## 每人必备（LR / RL 各一份）

| # | 内容 | 压缩包内路径（每 run） | 用途 | 校验 |
|---|---|---|---|---|
| 1 | 皮层 CIFTI | `Results/tfMRI_EMOTION_LR\|RL/tfMRI_EMOTION_LR\|RL_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii`；WM 同理 | cortex 400ROI 时间序列（`03_02`） | EMOTION 176 帧，WM 405 帧，`91282` grayordinates |
| 2 | 海马 volume | `Results/tfMRI_EMOTION_LR\|RL/tfMRI_EMOTION_LR\|RL_hp0_clean*.nii.gz`（或 `*.nii.gz`） | hipp volume→surface 采样（`02_01`） | 4D，第 4 维=176 / 405，与 CIFTI 帧数一致 |
| 3 | EV 事件文件 | EMOTION `EVs/fear.txt`（faces）、`EVs/neut.txt`（shapes），18s block；WM `EVs/0bk_{faces,places,tools,body}.txt`、`2bk_*.txt`，27.5s block | 一阶 GLM 设计矩阵 | FSL 3 列 `onset duration weight`，秒为单位 |
| 4 | 头动回归 | `Movement_Regressors.txt`（12 列）+ `Movement_RelativeRMS/AbsoluteRMS.txt` | GLM nuisance 协变量 + 质控剔除 | 行数=TR 数（176/405），列数=12 |
| 5 | 行为 | EMOTION `EMOTION_Stats.csv`（FACE/SHAPE 准确率/RT）；WM `WM_Stats.csv` 或 `TAB.txt` 类 | 脑行为关联校验，与组 CSV 交叉核对 | 与 `HCP_YA_subjects.csv` 对应列一致 |
| 6 | cue/正确性细分（WM） | `EVs/cue.txt`，`*_cor/_err/_nlr.txt` | 进阶：正确 vs 错误试次、cue 建模 | 可选，一阶先不用 |

Relational 如做对照，加一套同样的 `tfMRI_RELATIONAL_LR|RL`（232 帧）+ `match/rel` EV。

## 概念对照

* `*_EV/`：FSL 3 列 `[onset duration weight]`，GLM 设计矩阵核心。包内 `EVs/` 全有。
* `Movement_Regressors.txt`：每 TR 12 列刚体运动参数 + 导数，GLM 协变量。每个 run 都有。
* block 设计：同类刺激集中连续呈现（EMOTION 18s，WM 27.5s = 2.5s cue + 10 个刺激），`Duration` 列即 block 长，检测功效高。
* `func/` 只有 `task-rest_run-concat`：本地 BIDS `func/` 现只有静息拼接数据；任务包在 `/Volumes/Elements/HCPYA2025_TaskfMRI3T_preprocessed/`，需 stage（解压部署）到本地 `func/` 才可跑任务分析。

## 不用 stage 的东西

* `tfMRI_Concat/*`：14 任务大拼接，只适合跨任务连接组学，不进 GLM。
* `*_Atlas_MSMAll_hp2000_clean*`（如有）：rest 去噪版本线，任务用包内 `hp0_clean_rclean_tclean` 即可，不要混用。

## stage 落点建议

保持 HCP 原名，不要重命名，沿用现有布局：

```text
data/hippunfold_input/sub-<ID>/func/
  sub-<ID>_task-emotion_run-01.dtseries.nii  <- tfMRI_EMOTION_LR_Atlas...dtseries.nii
  sub-<ID>_task-emotion_run-01.nii.gz        <- tfMRI_EMOTION_LR volume
  sub-<ID>_task-emotion_run-02.*             <- RL
  sub-<ID>_task-wm_run-01/02.*
  .../EVs/、Movement_Regressors.txt 原目录结构保留
```

每人 stage 完跑三行校验：`nibabel` 读 dtseries/volume 帧数相等且等于 176/405；EV 为 3 列；`Movement_Regressors.txt` 行数等于帧数。任一项对不上就 fail，不进 `02_01`。

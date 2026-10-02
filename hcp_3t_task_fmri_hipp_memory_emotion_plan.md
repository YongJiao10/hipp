# HCP 3T Task-fMRI Hippocampus Memory/Emotion Analysis Plan

## 1. 现状：`scripts/core_workflow` 是 rest-only

* `01` HippUnfold T1w 解剖（`den-512`，任务无关，可复用）。
* `02_01`：4D BOLD volume 经 `wb_command -volume-to-surface-mapping -enclosing` 采样到 HippUnfold `midthickness`，见 `scripts/core_workflow/lib/surface_sampling.py`。run 级拼接只在内存，不落盘。
* `03_01/03_02`：CIFTI `*_Atlas_MSMAll.dtseries.nii` 皮层 grayordinates 按个体 ArealMSHBM `lh/rh_labels.txt` 做 `nanmean`，见 `scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py`。默认模板为 rest（`04_01` 内 `task-rest_run-concat.dtseries.nii`）。
* `04_01/04_02`：rest `corrcoef` hipp→cortex-400ROI / intrinsic FC。
* `05_01/05_02`：BrainSpace 梯度 + `shifted-cosine` 聚类，固定 `K=3`，`fisherz` 默认。
* 全树无 task / EV / onset / GLM / beta / contrast / 脑行为概念；`path_templates.py` 暂无 `{task}` 变量。

## 2. 目标：纳入 HCP 3T 任务态，做记忆/情绪 hippocampus 功能分析

优先 `EMOTION + WM`，`Relational` 作对照。`Gambling/Social/Language/Motor` 暂不进 hipp 主线。

* EMOTION：`faces > shapes`，杏仁核-海马反应性，情绪加工。
* WM：`0bk/2bk × face/place/tool/body`，编码负荷，长轴梯度位移。
* Relational：关系绑定，对照。

## 3. 数据输入

### 3.1 已有

* 解剖/HippUnfold：`data/hippunfold_input/sub-*/`（复用，不重跑 `01`）。
* 行为：`data/HCP_YA_subjects_2026_04_09_16_01_42.csv`，见 §5。
* rest 参考：现有 `outputs_hcp/.../routea` rest FC/梯度/`K=3 Anterior/Body/Posterior` 标签。

### 3.2 待 stage（当前 `func/` 只有 `task-rest_run-concat`）

每个被试、每个任务、每个 phase（`LR/RL` 即 `run-1/2`）：

* CIFTI：`tfMRI_{EMOTION,WM,RELATIONAL}_{LR,RL}_Atlas_MSMAll.dtseries.nii`（cortex 用）。
* Volume：`tfMRI_{...}_{LR,RL}.nii.gz` 或 `hp2000_clean.nii.gz`（hipp 采样用）。
* 设计：HCP `*_EV/`（onset/duration）、`Movement_Regressors.txt`。
* 质控：复用 Dice；tSNR 按 task 重算。

## 4. 提取链路：cortex 走 surface/dtseries，hipp 走 volume

与 rest 同构，只换输入源：

* cortex：task dtseries → 同一套 RouteA 个体 parcel labels → `cortex_roi_parcel_timeseries.npy`（task 版）。标签是解剖/个体化的，rest 标定的 labels 可直接用于 task。
* hipp：task volume → `volume-to-surface-mapping` → `den-512` hipp 时间序列。CIFTI 内 hipp `764/795` subcortical 体素不是 unfolded 拓扑，不能替代。
* `LR/RL` 按现有 run 契约处理：`run-artifact-manifest`（path/size/SHA256）+ `run-metadata`（run ID/length），内存拼接，缺 run 直接 fail，不回退 rest。

建议形式根（不覆盖 rest）：

```text
outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion/
outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-wm/
outputs_hcp/work/hippomaps_gradients/task-{emotion,wm}/
outputs_hcp/work/group_atlas/task-{emotion,wm}_full_k3_fisherz/
```

实现上：给 `lib/path_templates.py` 加 `{task},{phase}` 变量；`02_01/03_02/04_01` 加 `--task` 及 task 模板参数；manifest 记录 task/phase/run。

## 5. 任务指标（现成，无需新采）

来自 `data/HCP_YA_subjects_*.csv`：

* Emotion：`Emotion_Task_Acc/Median_RT`，`Face_Acc/RT`，`Shape_Acc/RT`。
* WM：`WM_Task_0bk/2bk_Acc/RT`，细分 `Body/Face/Place/Tool × Target/Nontarget` 的 `Acc/Median_RT`。
* Relational：`Relational_Task_Rel/Match_Acc/RT`。

另有完成度列（`fMRI_Emo/Wm/Rel_PctCompl`）、`FS_L/R_Hippo_Vol` 作协变量。

## 6. 分析三线

### A. 激活 GLM（先做：where）

* 对 hipp `den-512` 逐顶点一阶 GLM：HRF 卷积 + 高通 + 头动回归。
* 对比：`faces > shapes`；`2bk > 0bk`（可再分 stimulus）；`relational > match`。
* 组水平渲染到 folded/unfolded 表面，叠加 rest `K=3` 标签检验：情绪压前部、负荷调制长轴。

### B. 背景连接 / 梯度重构（how reorganize）

* 回归掉任务设计，用残差重跑 `04_01 corr + 05_01 BrainSpace` 同参数全套，对齐到 rest `4mm/hemi_L` 锚点，得 task 梯度库；比较 rest→task FC/梯度位移。
* 或 beta-series：逐 block 估计 → hipp-cortex 相关，看记忆/情绪条件耦合增强。
* 目的：把刺激诱发共激活与持续性网络重构分开。

### C. 脑-行为（performance）

* 单位：`cluster × contrast`（如前部 `faces>shapes beta`、后部 `2bk>0bk beta`、背景 FC 强度）。
* 模型：`Acc/RT ~ 脑指标 + FS_Hippo_Vol + 头动 + 家系结构`。
* 先验：前部-脸孔情绪准确率；体/后部-2bk place 负荷。

## 7. 实施顺序

1. stage task CIFTI/volume/EV 并核对 TR 数。
2. 加 `{task}` 模板 + task 共享根 + smoke run。
3. A 激活分支。
4. B 背景 FC/梯度分支。
5. C 脑行为模型。

## 9. 落地状态（2026-09-24，已实跑 sub-115017 emotion）

* 新编号（未放 `exploratory_beta`）：`04_03` 一阶激活 + `04_04` 背景FC；共享 `lib/task_design.py`（EV/HRF/设计矩阵/对比）+ `lib/task_glm.py`（OLS/残差）+ `lib/task_manifests.py`（manifest 生成器）。
* `--task` 输入口：`02_01`（task volume→surface 采样，逐 run 帧数校验 176/405）+ `03_02`→extractor（task dtseries→400ROI 参考，`reference_summary.json` 记 `task`）。`--task` 缺省 `rest`，rest 行为零改动。
* Manifest：`outputs_hcp/inputs/hcp/kong2022/manifests/task-{emotion,wm}/sub-*.json`，3 人 × 2 任务已生成（含 SHA256/size）。
* 实跑链（115017 emotion）：`02_01 raw/tsnr/smooth` → `03_02`（400×352 参考）→ `04_03`（`faces>shapes` 512 顶点图）→ `04_04`（512×400 背景FC），全绿。
* 已知问题：rest tSNR 门限（invalid fraction 0.20）对 task 太严——115017 emotion L/R invalid 为 0.91/0.82，主因是短 run + `hp0_clean` 尺度差异。task 需要独立 tSNR/QC 阈值（待定），pilot 的 smooth 是绕过 QC 直调 `run_smoothing` 做的。
* Pilot 残留：`outputs_hcp/inputs/hcp/kong2022_task_emotion/`（route-root，内有空 QC 名单），复用前需清理或重跑 QC；删前需按规则先确认。

## 10. 阶段一结果（N=25，2026-09-25）

* 组构建 `05_06`：roster 全自动发现（emotion/wm 各 25 人），稳定路径 `work/task_activation/task-{emotion,wm}/group/`，roster 指纹记 manifest；渲染 `06_11`：组表面 overlay + cluster bar + 长轴 profile，落 `present/task_activation/`。
* 关键坑：`group_labels_k3.npy` 是任意 k-means id，必须用 `group_labels_final.npy` + `chosen_k_final` + `cluster_order_head_to_tail` 配套（见 `lib/task_group.load_group_labels`）。
* faces>shapes：Anterior（t≈3.1-3.5）/ Body（t≈4.2-4.3）激活，Posterior 弱（t≈1.5-2.1）——Body 最强，非纯前部效应。
* 2bk>0bk：Body 显著去激活（t≈-3.3 到 -4.3），Anterior/Posterior ≈ 0——与情绪任务方向相反。
* 增量：新人传完后重跑 `05_06` same 命令即可，N/指纹自动更新，无硬编码人数。

## 11. 双分支分区 + 面积加权 + 单图汇总（2026-09-25）

* 分区命名沿 JJ_TMS 惯例：`hcp151_atlas`（组 rest K=3）与 `template_matching`（crossfit 个体 K=3），表格里 `parcellation` 列区分。
* 聚合改为 native 面积加权 only（无双报告）：`subject_cluster_rows` 用每个人自己的中层表面顶点面积加权，列为 `mean/sd/area_mm2`。面积函数收拢进 `lib/surface_utils.vertex_area_weights`。
* 个体标签使用前逐人校验头→尾单调（用其 native 表面轴），不过则显式回退组标签并记 `parcellation_note` + manifest `parcellation_fallbacks`。本轮 12/100 人次回退。

## 12. 去校验（2026-09-25）：TM 是功能身份标签，不要求空间单调

* 删 `validate_individual_labels` 及回退逻辑：个体 2 号damn功能像模板 2 号（Body），无论落在哪；位置漂移是信号不是噪声。
* 缺 TM 文件才剔除（仅该 zone，记 `excluded`），组 zone 保持全员。
* 实测：template_matching zone 更灵敏——faces>shapes Body t 4.53→5.08，Posterior 1.57→2.92（转显著）。

## 13. beta 入图 + 2mm only + 分开展示（2026-09-25）

* 只做 2mm：`05_06/06_11` 缺省改单 smoothing；4mm 旧产物留盘不动。
* `05_06` 加各条件 beta 组图（`group_beta_{stem}_*`）与 beta cluster 行（contrast 列记 `beta_{stem}`），附双尾 p（`group_p`）。
* `06_11` 每 task 两张图：`_profile.png`（各 beta + contrast 长轴线，95% CI 带）、`_bars.png`（bar=组均值±95%CI + 被试散点 + 显著标记 `+/*/**/***`）。
* 关键所见：faces>shapes 对比由“脸激活 + 图形去激活”双驱动（beta_neut 为负），减法图 alone 会丢一半故事。

## 14. 一图汇总 + 去CI + 2mm无folder（2026-09-25）

* 每 task 一张 `task_{task}_{contrast}.png`：上行 profile 线（各 beta + contrast，无 CI 带），下行 3 bar（contrast，hcp151_atlas，散点 + 95%CI 须 + 标记）。
* CI 带去掉——CI = mean ± t×SEM（约±2.06 SEM），被试间差异大时带子太大喧宾夺主；bar 上的须保留（细线，不占面积）。
* present 下 overlay 直接放 task 根，不建 `2mm/` folder。template_matching 只留表里，图上只画 canonical 的 3 bar。

## 15. 一图三问全解（2026-09-25）

* profile 按对比正负权重组并线：WM 只画 3 条（beta 2bk四条件均值 / beta 0bk四条件均值 / contrast），9 线意大利面删除。
* bars 只画 contrast，但双 zone 并排（每半球 6 bar：3 分区 × hcp151/tm），marker 可直接对比哪个分区更优。
* present 下无 smoothing folder，overlay 直放 task 根。

## 16. 图细节四改（2026-09-25）

* 标记智能放：正 bar 放上沿、负 bar 放下沿，不再无脑置顶；subregion 两两配对 t 检验（同被试配对）显著才画横线 bracket，分层避叠。
* 名称统一：`twobk_gt_zerobk` 显示为 `2bk>0bk`，beta 显示 `2bk/0bk`；分区显示 Ant/Mid/Post（数据里仍是 canonical 全名），zone 只靠 legend 颜色区分，x 轴不加后缀。
* 散点缩小（s=9，窄 jitter，alpha），为 151 人规模预留。

## 17. GPT review 整改（2026-09-25，R01 preliminary导向）

Standards（7项全改）：
* P1 无效零值：复核属实（中位~370/512全零==tsnr invalid精确对应）。`04_03/04_04` 现载 mask 置 NaN，contrast/bgFC 零 exact-zero，coverage（中位0.21）记 summary/表/图。
* P1 HRF：按 SPM 官方 `spm_hrf.m`+`spm_Gpdf.m` 重写（t^5·e^-t 核，峰 5.04s，sum归一），回归测试锁定。
* P1 背景FC不对称：双侧回归 + NaN 传播相关（`corrcoef_rows_nan`，缺失永为NaN）；定性为 exploratory，未接 `05_01`。
* P2 run 配对：`04_03/04_04` 强制 `--task-manifest-root`，run 身份/顺序/长度全从 manifest + volume 头读；另存 per-run 对比（跨 run 可重复性备用）。
* P2 roster 指纹删除，只留名单 + 输入文件 SHA/size。
* 一阶设计仅截距+线性漂移（无高通、无自相关处理），block 设计下可接受，正式使用需声明。

Spec（5项）：
* 主分析改为直接的分化增益检验：`differentiation_gain.tsv`（配对 diff-of-diffs，FDR），zone 扩展三分支 + 解剖三分 comparator。结果：FDR 后无线——个体化更优**暂无直接证据**（单样本 t 高≠分化强）。R2 已改名 `zone_diff_change.tsv` 并去优越性表述（见 §19）。
* 星号/bracket 全切 FDR（组检验一个 family，bracket 图内一个 family）。
* 亲属：25 人含一对 MZ（126426/146735）；24 人敏感性结论不变。

## 18. 每 task 一张大图（2026-09-25）

* `task_{task}.png`：纵向堆（每 contrast 占 3 行 profile/bars/change × L/R），WM 一张 9 行全览；8 个旧 per-contrast PNG 已删（经确认），每 task 目录只留一张大图。
* beta/contrast 统一名单（同 combo 缺一即剔除，记账）。
* TM 更灵敏的概括收回；脑行为列下一阶段。

## 19. 第二轮 review 整改 R2（2026-09-26；当时未重跑，现见 §21）

Standards（3项）：
* R2-1 run顺序核对：`lib.task_manifests.verify_cached_runs_against_manifest`，`04_03/04_04` 用 `surface/raw/surface_sampling_summary.json`（+`04_04` 另验 cortex `reference_summary.json`）逐 run 核对 manifest 的 run id/长度/源路径，等长 LR/RL 换序必报错；校验收据写入 output summary。
* R2-2 显式名单缺失 fail-closed：`05_06` 显式 `--subjects/--subjects-manifest` 缺 beta/contrast 直接报错点名（缺谁缺哪个文件），只自动发现模式允许记账排除。
* R2-3 FDR 回退删除：`06_11` `group_p_fdr`（bars）/`delta_p_fdr`（change 行）缺失直接报错，不再回退 raw p。

Spec（3项）：
* R2-4[P1] 分区覆盖门限：`subject_cluster_rows` 新增 `--min-valid-area-fraction`（默认 0.10），有效组织面积不足者 NaN + `excluded_low_coverage`，排除明细入 `coverage_exclusions.tsv` + manifest；实测分区有效比例中位仅 0.1–0.3，0.25 门限将排除约半数 subject-zone（敏感性对比 0.10 vs 0.25 待重跑后出）。
* R2-5 load profile 权重修正：side 曲线改权重归一加权（pos−neg 精确重构 contrast，review 反例 0.5 已回归测试锁定），等权对比行为不变。
* R2-6 去“增益”优越性表述：`differentiation_gain.tsv` → `zone_diff_change.tsv`（`delta_*` 列），图行改名 zone-diff change + 双尾注释（符号是方向非优越），图注同步。

单测 39 passed（含 7 个新增：receipt 匹配/换序/长度漂移/源漂移/缺失、覆盖门限、权重重构、FDR 必填）。

**样本更新**：新被试到来，当前共 34 人。上述代码改完后**未重跑**（磁盘上仍是 25 人结果），待 34 人 `02_01→03_02→04_03→04_04` 就绪后一次性重建 `05_06/06_11`（含 0.25 敏感性 + 24/25/34 相关结论复核）。

可视化（R01 prelim 版）：Emotional faces/Shapes 命名；beta 单位 a.u.；双半球同 y 轴；panel 标 N/coverage/校正；bin 轴脚注（PCA 等顶点数非等距解剖）；`2bk<0bk` 表述为相对更低非去激活。

待定（需你拍板）：task 专用 QC 阈值重标定（当前 coverage 0.05–0.50）；背景FC/梯度转正前评估灵活响应基。

* 标记智能放：正 bar 放上沿、负 bar 放下沿，不再无脑置顶；subregion 两两配对 t 检验（同被试配对）显著才画横线 bracket，分层避叠。
* 名称统一：`twobk_gt_zerobk` 显示为 `2bk>0bk`，beta 显示 `2bk/0bk`；分区显示 Ant/Mid/Post（数据里仍是 canonical 全名），zone 只靠 legend 颜色区分，x 轴不加后缀。
* 散点缩小（s=9，窄 jitter，alpha），为 151 人规模预留。
* `06_11` 每 task 一张汇总图（`task_{task}_{contrast}_summary.png`，2 smoothing × 2 hemi，profile + 双 zone cluster 点）；旧 per-combo PNG 只上报不删除。
* 个体化 zone 的组 SD 明显大于组 zone（个体地盘每人不同，符合预期）。

## 8. 合规提醒

* 缺 task run 必须 fail closed，禁止静默回退 rest（No Silent Fallbacks）。
* 精确 artifact 选择；日期/run ID 只进 manifest/log，不建新日期结果树。
* 不在 `scripts/core_workflow/` 根加无编号脚本；共享逻辑进 `lib/`；改行为必同步文档。


## 20. 绘图修复收尾（代码与文档准备，不更新结果）

* 条件 beta 合成保持固定权重；任一必需条件非有限时，该顶点结果为 NaN，禁止对剩余条件重新归一化。
* 区域柱逐柱显示统计表中的有效 n；跨表示差异柱逐柱显示完整配对 n。图总标题的 cohort N 仅表示队列总人数。
* 回归测试覆盖单条件缺失、全部缺失、非有限输入、完整条件加权，以及区域排除后和配对缺失后的 n 标注。
* 按用户安排保留现有结果，待新被试就绪统一重跑；本次收尾仅涉及代码、测试和文档。


## 21. 34 人正式 task 根原位重建完成

本轮按用户授权覆盖 EMOTION / WM 的现有 2mm 结果：34 人全部重跑
`04_03` 一阶激活（含 LR/RL 单 run 对比）与 `04_04` exploratory 背景 FC，
随后以显式同一 34 人名单重跑 `05_06`、`06_11`。10% 有效组织面积门限、
三种分区表示、BH-FDR 和当前一阶设计保持既定参数。

输入位置明确分组，未采用按存在性择路：旧 25 人的 surface 根为
`outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-<task>/surface/`；
新增 9 人（162935、164636、165436、167036、167440、169343、169444、169747、171633）
的 surface 根为 `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-<task>/`。
各人 source/run 顺序及长度均由现有采样记录与 task manifest 核对。

当前使用的组统计为 `outputs_hcp/work/task_activation/task-<task>/group/` 内
`subject_cluster_table.tsv`、`group_cluster_summary.tsv`、`coverage_exclusions.tsv`、
`zone_diff_change.tsv` 与 `group_build_manifest.json`。
当前图为 `outputs_hcp/present/task_activation/task-emotion/task_emotion.png`
和 `outputs_hcp/present/task_activation/task-wm/task_wm.png`，均原位覆盖。
人数标签移到绘图区上方，paired n 分两行显示，避免重叠。

结果核查：每个任务 68 个 subject×hemisphere 一阶与 FC 输出均通过 run 核对；
无效顶点保持 NaN；run 平均与合并 contrast 一致；区域覆盖不足者不进入统计。
EMOTION 的 contrast 区域有效 n=20–34、跨表示完整配对 n=19–32；
WM 分别为 n=18–34 和 n=18–29。三个表示合计的 contrast 区域检验
经当前 family 的 BH-FDR 后，EMOTION 14 项、WM 4 项小于 .05；
两任务跨表示 contrast 差异变化均无 BH-FDR 显著项，不能声称个体化优越。

历史 `4mm/` 与旧 `differentiation_gain.tsv` 未被本轮重建或删除，不能用作本轮
34 人结果；当前入口消费 2mm 与 `zone_diff_change.tsv`。本轮没有执行删除操作。
25% 覆盖率和去亲属敏感性分析仍待单独完成，旧 25 人敏感性结论不外推为34人结论。

### 2026-09-26 task validity correction (code and approved 2mm products rebuilt)

This supersedes earlier statements that task QC still uses the rest tSNR gate.
Task hippocampal and cortical preparation now use finite, nonconstant signal
in every declared run, with no total-SD cutoff. Existing gate filenames remain,
but task summaries declare `finite_nonconstant_each_run` and null tSNR threshold.
Old task smoothing/reference metadata are rejected by first-level consumers.
See `scripts/core_workflow/README.md` for the ordered rebuild commands.
A read-only preview for sub-115017 gives EMOTION L/R 512/512 basic-valid vertices
(previously 46/91) and WM L/R 512/512 (previously 51/63). This is signal validity,
not evidence that all vertices meet acquisition, motion, or model quality criteria.

User approved production replacement in these six exact directory scopes,
relative to `/home/yojiao/hpchome/hipp/`:

- `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-emotion/`
- `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-wm/`
- `outputs_hcp/work/task_activation/task-emotion/`
- `outputs_hcp/work/task_activation/task-wm/`
- `outputs_hcp/present/task_activation/task-emotion/`
- `outputs_hcp/present/task_activation/task-wm/`

Replace derived task masks, smoothing, cortex references, task QC lists, individual
activation/background FC, group activation and presentation outputs in place.
Use the existing task run manifests and an explicit roster with available inputs.
Keep raw sampled timeseries as inputs; rebuilding starts at the signal-validity
stage. This list authorizes no deletion and no rest/template-tree replacement.

#### Approved rebuild completed

EMOTION and WM were rebuilt for the existing explicit 34-person roster on IDAS
with two subject workers and native math threads limited to one. Both tasks
completed hippocampal masks/smoothing, cortical references, 2mm activation and
background FC, group statistics and presentation rendering. The existing
25-person surface root and documented additional-9 root were selected explicitly.
Dice and invalid-vertex-fraction QC passed for all 34 subjects in each task.

Direct output checks covered 68 subject-hemisphere activation/FC products per
task: FC shape 512 x 400, finite contrasts/FC, recorded mask coverage, and
agreement of saved run means with combined contrasts. Each task now has
34816/34816 basic-valid hippocampal vertices across 34 people. Group coverage
exclusion entries changed from 237 to 0 (EMOTION) and 814 to 0 (WM).
These are region/statistic entries, not subject counts. Basic signal validity
is distinct from a claim of perfect acquisition quality.

Current figures are `outputs_hcp/present/task_activation/task-emotion/task_emotion.png`
and `outputs_hcp/present/task_activation/task-wm/task_wm.png`; both were visually
inspected after rendering. Stage logs are under each approved task shared root
at `logs/validity_rebuild_20260926/`. Maintained downstream products are 2mm;
legacy 4mm first-level/group products were not included in this rebuild.

## 2026-09-26 actual tSNR correction

The interim finite/nonconstant-only gate was replaced with per-run original
mean / task-model residual SD >=25, intersected across declared runs. HCP7T
rest is unchanged. Within the six already approved task roots, all 34 individual
EMOTION/WM products were rebuilt at 2mm and 4mm. Retaining the existing <20%
invalid-vertex and Dice criteria gives final group N=13 for EMOTION and N=11
for WM (the previous validity-only groups had N=34). Updated group tables
contain both smoothings; the maintained figure explicitly selects 2mm.
Detailed before/after counts are in
`docs/tsnr_cross_dataset_assessment_2026-09-26.md`. Threshold 25 is inherited,
not a newly validated universal residual-tSNR cutoff.

## 2026-09-28 task figure simplification (two zones, dual brackets)

Per user approval: `anatomical_thirds` branch removed from `05_06`/`06_11`/`lib/task_group`
(two zones remain: `hcp151_atlas`, `template_matching`); `zone_diff_change.tsv`
is no longer written (stale copies under `work/task_activation/task-*/group/`
are legacy until explicitly cleared); the zone-difference figure row is removed
(`n_rows` per contrast 3->2). Cluster bars carry paired subregion brackets
(black, long end caps, one BH-FDR family; method-contrast brackets removed
per user request). Profile bins are numbered 1-20
(1 = head/anterior). Figure PNGs re-rendered at 300 dpi with Arial/Helvetica.

## 2026-09-28 new subjects 55-63 ingested (N=54->63)

Source root `HCP_tast_3T` holds 63 subjects; 9 were missing from
`shared_inputs` (`197348,199655,200210,200311,200614,201515,203418,204521,205220`).
Built 18 run-artifact manifests, ran `02_01` surface sampling+gating+smoothing
(`tsnr-threshold none`, matching production) for both tasks, independent QC
(`qc/new9_validity`, 9/9 pass validity-only, existing `validity_only_54`
untouched), and `04_03` first-level activation (2mm+4mm). Group tables and both
figures refreshed at N=63 with zero coverage exclusions.

## 2026-09-28 WM 9-contrast expansion, physiological profile decomposition, and 2x4 category layout

Per user request:
1. Working memory contrast suite expanded from 3 to 9 in `lib/task_design.py`:
   - Cognitive load main effect: `twobk_gt_zerobk`
   - Category specialization (4): `faces_gt_others`, `places_gt_others`, `tools_gt_others`, `body_gt_others`
   - Cognitive load × category interactions (4): `load_x_faces`, `load_x_places`, `load_x_tools`, `load_x_body`
2. First-level activation materialized across all 63 subjects (`04_03`) and group cluster tables built (`05_06`) for all 9 contrasts at 2mm.
3. Profile line decomposition refactored with physiological meaning (`contrast_physiological_sides` in `06_11`):
   - Category contrasts: Target category mean (0.5 * 0bk + 0.5 * 2bk) vs remaining 3 categories mean (1/6 * 0bk + 1/6 * 2bk each).
   - Interaction contrasts: Target category load effect Δ (0.5 * 2bk - 0.5 * 0bk) vs remaining 3 categories load effect Δ (0.5/3 * 2bk - 0.5/3 * 0bk each).
   - Load contrast: 2-back mean (0.25 * 2bk across categories) vs 0-back mean (0.25 * 0bk across categories).
   In all cases, `Line 1 (blue) - Line 2 (orange) = Contrast (black)` holds algebraically on complete vertex support.
4. Two dedicated presentation figures rendered (`06_11`):
   - `task_wm_load.png`: Standalone 2x2 figure for cognitive load (`2-back > 0-back`).
   - `task_wm_categories.png`: High-tier 2x4 card layout grouping the 4 categories (Faces, Places, Tools, Body) across Category Specialization (Row 1) and Cognitive Load Interaction (Row 2), with category-specific color-coded badges, visual card dividers, paired-subregion brackets with figure-wide BH-FDR correction, and row-wise unified Y-axes for both profile curves and cluster bars to enable fair, direct horizontal comparison.
5. Significance bracket collision avoidance & frame headroom overhaul (`06_11`):
   - Hierarchical span-length sorting: Shorter-span brackets (adjacent subregions) are evaluated and placed first, followed by longer-span brackets (Ant-Post).
   - Global interval stacking per axis: Eliminates per-zone level reset collisions between `hcp151` and `tm`; horizontally overlapping brackets receive distinct vertical tiers with >=9.0% span clearance, preventing horizontal lines from intersecting lower asterisks.
   - Dynamic top headroom expansion: Expands the upper y-limit by >=11-14% of span above the highest marker, guaranteeing that asterisks never collide with the top border or overflow into the n=63 count annotations.



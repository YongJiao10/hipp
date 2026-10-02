# tSNR QC 事件交接 — 2026-09-26

## 当前用户决定（优先于此前方案）

用户明确要求：先参考权威文献；当前不做任何筛选，至少等 HCP 3T task 的 151 人数据完整后再判断。暂停新增顶点/run/被试筛选、阈值部署、队列调整及由此触发的下游重跑。不要继续此前建议的针对性筛选验证。数据完整后先报告完整性并与用户确定恢复工作，不能自动启用 25、20 或 15。

本次交接只记录暂停和文献依据；没有回滚、删除或重建现有生产结果，也没有把历史掩膜改成全通过。现有源码仍包含筛选逻辑，不能把“用户决定暂停”误解为“代码已经关闭筛选”。在继续数据收集/准备时，不能直接启动会应用这些门槛的全流程。

## 事件与判断

- 起因：HCP 7T rest 的 `10000/std >=25` 被沿用到 Stanford/Iowa 的 XCP-D 残差以及 HCP 3T task；task 大量失败。
- 确认的问题：XCP-D 残差没有 HCP 的固定 10000 基线依据；task 拼接不同 run 后直接算 SD 会引入 run 均值差异，并混入任务相关方差。
- 前次修正：HCP7T保留原规则；其他 rest 用保留基线的每run均值/去线性趋势SD；task 用每run均值/任务模型残差SD。额外组合了“每run >=25、所有run取交集、任一海马半球invalid >=20%淘汰被试”。
- 独立复核结论：新的计算定义有方法学依据，但 **25 + 所有run交集 + 20%被试门槛** 没有在本项目得到验证。算法能运行及重现掩膜，不等于筛选标准已被验证。
- 撤回此前“Stanford/Iowa下游必须全部重跑”的无条件结论。准确说法：只有最终采用不同分析掩膜/时序/队列时，才需更新受影响的依赖结果。新掩膜改变FC不证明它必然更正确。
- 此次没有发现必须修改 HCP7T rest 的依据；这也不等于已经证明25是通用科学边界。

## 当前磁盘状态

| 分支 | 已发生的变化 | 接手时处理 |
|---|---|---|
| HCP7T rest及参考模板 | 本次tSNR修正未重建 | 保留 |
| Stanford36P/22P、Iowa3T JJ_TMS36P/22P | 已做候选mask/FC影响审计；正式rest结果未替换 | 保留，不启动全面重跑 |
| Iowa7T | 海马候选规则比较；正式结果未替换 | 保留 |
| HCP3T EMOTION、WM | 此前每任务34人已按25规则重建mask相关个体结果；group仅13/11人 | 视作待定规则下的结果，不能代表最终151人QC；暂停进一步筛选/重建 |

用户此次没有要求恢复task旧mask或生成全通过mask，所以没有执行此类覆盖。当前“34人”来自既有审计，不是本次重新扫描151人收集进度的结果。

## 已有证据，避免重复全量审计

主要报告：`docs/tsnr_cross_dataset_assessment_2026-09-26.md`。
审计notebook：`docs/tsnr_existing_mask_scope_2026-09-26.ipynb`。
文献补充：`docs/tsnr_hcp_primary_sources_2026-09-26.md`（本次研究任务写入）。

保持其他规则不变时，task通过人数：

| 阈值 | EMOTION | WM |
|---|---:|---:|
| 15 | 34/34 | 34/34 |
| 20 | 27/34 | 30/34 |
| 25 | 13/34 | 11/34 |

这些是敏感性结果，不是对任何阈值的推荐。15时全部通过不自动意味着QC无效。25时剔除很多人也不自动意味着被剔除者数据不可用。

rest在候选25规则下的名单变化：Stanford36P 194→180、Stanford22P 192→180、JJ36P 94→82、JJ22P 94→87。每个审计对象的皮层mask至少一个顶点变化；这是“采用该mask后的依赖变化”，不是部署理由。

特别注意：JJ22必须用trilinear，与JJ36的enclosing-voxel不同。早期JJ22 94→82结论错误，已更正为94→87。Stanford36阈值扫描195人未包括单独计算的4117；4117原已因coverage失败，不能作为新增淘汰。Stanford单run没有多run交集损失。

Iowa7T比较：11个可计算产品中5个multiband在25/20%组合下失败，6个multi-echo通过；另2个产品指定baseline缺失。multi-echo baseline在tedana后、XCP-D前，multiband在XCP-D前，两者预处理阶段仍须明确。Iowa7T皮层未完成重算。

FC pilot只覆盖7个例子，固定旧海马时序、改变皮层mask；不等于完整新海马平滑+FC的效应，也没有验证最终科学结论是否改变。

## 源码入口及恢复注意事项

- `scripts/core_workflow/lib/tsnr.py`：每run均值/残差SD、ddof=1、默认25、所有run取交集。
- `scripts/core_workflow/02_01_prepare_group_inputs.py`：海马mask及后续平滑入口，25仍写在流程中。
- `scripts/core_workflow/02_02_build_qc_manifest.py`：任一半球invalid达到20%淘汰（并有Dice要求）。
- `scripts/core_workflow/lib/io_utils.py`、`lib/extract_cortex_kong2022_route_a_roi_timeseries.py`、`03_02_build_kong2022_routea_reference.py`：皮层mask/ROI入口。
- `scripts/core_workflow/lib/prepare_tsnr_baseline_inputs.py`：显式baseline及保留帧打包。
- `scripts/jj_tms/jj_tms_hcp151_fixed_template_matching.py`、`scripts/iowa7t/run_branch_downstream.py`、`run_runwise_downstream.py`：独立入口也已接入新规则。

未来执行前必须先读当前源码，不能凭历史stage编号或报告行号直接启动。工作区存在其他未提交修改，不能整体reset/revert。历史95项测试针对过渡期finite-only规则，不验证后来的真实tSNR代码。

审计使用了 `/tmp/full_rest_tsnr/`、`/tmp/jj22_trilinear_tsnr_results.json`、`/tmp/stanford_4117_tsnr_results.json`、`/tmp/rest_cortex_fc_effect.json` 等临时缓存。它们可能消失；notebook记录了计算过程。缺缓存时报告缺失，不静默替代输入，也不擅自启动全量重算。

## 等151人完整后再恢复的工作

1. 确认每任务、每run真实数据完整性和处理版本；不能把文件存在等同于完整可用。
2. 先读权威文献，明确指标计算阶段、空间（海马512表面与全脑/皮层不同）、是否含任务回归及run处理方式。HCP经典文献不自动适用于后来的clean/rclean/tclean发布版本。
3. 向用户提出有文献和完整数据依据的dataset-specific规则；不要按通过率或目标显著性反向定阈值，也不要仅因用户要求dataset-specific就给不同队列任意不同数字。
4. 规则确定后再列出实际变化及最小依赖重算范围。修改mask不要求重跑fMRIPrep、XCP-D或HippUnfold。
5. 遵守当前AGENTS：正式结果树覆盖需该操作的精确路径批准；此前task批准不自动授权Stanford/Iowa覆盖；禁止未经批准删除及访问archive。

本次暂停期间只允许本次授权的文献研究和交接文档整理，不继续筛选、门槛试验或下游重建。

# HCP ctxreg / ctxdrop Closeout Implementation Plan

> **For Luna Max / agentic workers:** 使用可用的 executing-plans 技能逐项执行；若技能不可用，按下面检查点顺序执行并记录证据。不要重复启动已有计算。本计划只交接，尚未执行。

**Goal:** 不改变已经选定的科学定义，完成 ctxdrop 注释标识、正式路径 provenance 核验、六张 overview 验收及 Git 交付。

**Architecture:** 将控制后空间分区和用于解释分区的 FC 注释明确区分。修复展示语义与可追溯信息；仅在证据显示数值或依赖失效时重算受影响的下游，不以连接值大小判断正确性。

**Tech Stack:** Python py314、现有 core_workflow、Matplotlib、NumPy、pytest、Git。

**Spec:** 用户本轮明确选择“保留原始 FC 注释并明确标注”；参照 `docs/kong400_fc_contract.md`、仓库 AGENTS.md。本计划位于已有 docs 目录，不创建新目录。

## 1. 不可变的科学口径

| 产品/步骤 | 必须保留的定义 |
|---|---|
| 标准 RouteA | 对照分支；本次不改算法、不重建、不删除 |
| ctxreg | 每个海马顶点按 run，将 20 mm 内局部皮层平均信号及截距从海马时间序列回归掉；残差用于控制后 FC 与 cluster 注释 |
| ctxdrop 分区/WTA | 每个海马顶点分别排除其局部皮层邻域，再构造对应 FC 特征；不改成 cluster 共同邻域 |
| ctxdrop gradient Fisher-z 注释 | 控制后得到的空间 cluster，使用原始海马及标准皮层 ROI 时间序列计算注释；必须显式标明原始 FC |
| ctxdrop gradient normalized_top20_count 注释 | 同一控制后 cluster，计数来自原始 FC 排序；不是“排除后的 cluster FC” |
| cluster 注释统计 | `cluster_mean_timeseries`：受试者内先平均 cluster 海马时间序列，再与 400 ROI 相关并 Fisher-z，随后组平均、网络内 ROI 等权汇总 |
| 禁止恢复 | `vertex_fc_fisherz` 作为生产模式；cluster 全顶点邻域并集；新归一化/阈值以抬高 FC；缺失输入时偷偷读取别的缓存 |

旧图约 0.13 与当前约 0.04 的既有调查结论是注释信号来源不同，不仅是 8net/7net 差异。既有同标签、同 7net 比较：2mm 左侧 C3 DMN，原始信号 0.1328，残差信号 0.0379。这是此前核对证据，不是本次全量重新验证，不设为硬编码验收阈值。Lynch 的局部回归逻辑支持回归方向，不提供这里数值大小的直接金标准；不能将全部被回归方差都断言为伪影。

## 2. 交接现场：已知与未完成

2026-09-13 只读现场确认：

- 工作区 `/home/yojiao/hpchome/hipp`；主机此前识别为 IDAS，执行者仍需重新识别。
- PID `1652039` 的 `04_00_materialize_cortex_regressed_surface_timeseries.py` 仍存活，核对时 elapsed 34:05，`--jobs 8`。
- 已有运行命令如下，**仅供识别，禁止直接再执行一遍**：

```text
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py --source-shared-root /home/yojiao/hpchome/hipp/outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --source-surface-root /home/yojiao/hpchome/hipp/outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --out-shared-root /home/yojiao/hpchome/hipp/outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg --jobs 8
```

- 原 agent 会话 ID 45192 未必能跨 agent 使用。进程结束不等于成功，应检查日志和完整产物。此前没有保存全队列重建前的数值哈希，不能事后声称已经证明全队列重建前后完全一致。
- 先前元数据审计发现控制输入有已不存在的 `kong400_regeneration_20260911` 输出路径引用；运行中的重建尚未完成。ctxreg FC 层还需审计。不要为了路径字符串直接把两条分支全量重跑一遍。
- 六张 overview 已存在，但 `06_01` 尚无本轮要求的 ctxdrop 原始 FC 标注。不能把文档已有解释当成图已修复。
- 先前检查报告过 16 个 group manifest 为 ok/N151；本次最终交付必须再核验。

现有 dirty files（不要覆盖或整批提交）：

```text
M docs/kong400_fc_contract.md
M docs/rendering/wb_locked_rendering.md
M scripts/core_workflow/05_02_run_group_atlas.py
M scripts/core_workflow/06_01_build_group_overview.py
M scripts/visualize/README.md
M tests/core_workflow/test_group_overview_affinity_provenance.py
M tests/core_workflow/test_kong400_subject_completeness.py
?? pending_ctxprofile_deletion_20260913.txt
?? scripts/visualize/hcp_cortex_control_gradient_wta/
?? scripts/visualize/template_matching_dominant_subregions/
```

## 3. 执行安全边界

- 不访问 archive，不重复删除，不创建 dated/recovered 正式结果树，不动标准 RouteA。
- 本计划不是新的批量删除或数据重建批准。发现需要删除/替换结果树时，给出精确路径清单并等待确认；纯展示标注按用户选择在现有正式产品内更新。
- 修改文件用 apply_patch；尊重其他 agent 修改。历史参考文字与运行输入分开，不全仓盲目替换日期字符串。
- 先检测 SYSTEM_NAME/CLUSTER_NAME/SGE_CLUSTER_NAME/hostname。IDAS 总并发上限 10，含现有运行任务；不要再开第二组八进程。先核查父进程及 BLAS 线程，后续命令设置三个线程环境变量为 1。
- 遵循 AGENTS.md 的目录读取限深和 timeout 约束；遇超时停止，不无界递归扫描共享存储。
- Python 环境信息先按仓库要求打印 `python -c "import sys; print(sys.executable)"`，实际任务全部用 `/home/yojiao/.conda/envs/py314/bin/python`。

## Task 1：接管与冻结证据

- [ ] 执行 `git status --short`、`git diff --stat`，逐文件阅读本任务相关 diff，记录 HEAD。
- [ ] 用 `ps -p 1652039 -o pid,etime,stat,args` 核实 PID 是否仍是上面的命令；检查其工作进程和线程数。PID 可能复用，不能仅按数字操作。
- [ ] 在上游写入结束前，不渲染或重算其依赖产品，避免读取半更新数据；可以先写测试和标注代码。
- [ ] 从两个正式 overview manifest 枚举六张图及其 group manifest 输入，不凭文件名猜实际数据来源。
- [ ] 记录现有 cluster labels、主 profile 数组、K/ARI 和比例数据的哈希，作为此次“仅改标签”前后不变证据；不要宣称它代表之前 04_00 重建前的状态。

**验收：** 有明确运行状态、六图清单、现有修改归属和数值基线，无重复计算。

## Task 2：实现明确标注及结构化 provenance

**修改：** `scripts/core_workflow/06_01_build_group_overview.py`。

**测试：** `tests/core_workflow/test_group_overview_affinity_provenance.py`，保留其现有 affinity/layout 修改。

- [ ] 先增加测试：ctxdrop + gradient-spectral 的 fisherz 与 normalized_top20_count 都获得原始 FC 说明；标准与 ctxreg 不获得该说明；WTA 不被误标为 cluster 原始 FC 注释。
- [ ] 在现有 provenance 构建流程增加小型、可测试的注释语义 helper。建议输出字段：

```python
{
    "cluster_annotation_signal": "original_timeseries",
    "cluster_annotation_note":
        "Original-FC annotation of cortex-controlled clusters",
}
```

- [ ] 字段只在实际源 manifest 与该语义吻合时生成：核验 `profile_fc_mode`、实际 profile surface/reference 来源及四个 smoothing/hemi 输入的一致性。若源字段不足以证明来源，先追踪其有效配置与文件，不按路由字符串直接断言；不匹配时给出包含 manifest 路径和实际值的错误。
- [ ] 在 `render_combo` 的实际 PNG 增加清晰副标题/注释：`Original-FC annotation of cortex-controlled clusters`。normalized count 再明确 `Counts derived from original-FC ranking`。调整顶部留白，不能遮挡主标题或 heatmap。
- [ ] `overview_manifest_item` 传递新字段，`write_readme` 将含义写入生成的 README。不要仅改标题变量而不写入图像，也不要只改 PNG 而漏 manifest。
- [ ] 测试必须覆盖字段从 render metadata 进入 overview item 的保留，以及混合/不匹配 provenance 的拒绝。
- [ ] 执行新增测试看到合理失败，再完成最小实现并运行通过。不要重构整个绘图系统。

**验收：** 两张 ctxdrop gradient 图的含义可见、机器可读；WTA 和 ctxreg 含义未混淆；数值计算没有变化。

## Task 3：结束上游重建并核验正式路径

**检查对象：** 两个控制分支 `outputs_hcp/inputs/hcp/kong2022/shared_inputs/<route>/` 内 04_00/04_01 summary、stage manifest 及被下游实际消费的路径。

- [ ] 等待已有 04_00 正常结束，或记录失败及具体 subjects。根据其真实目录结构和 subject manifest 枚举 151 subjects × 2 smoothing × 2 hemi = 604 个预期 regression 单元，不使用无范围全盘扫描。
- [ ] 检查每个单元 residual、summary、manifest 存在且可读；状态、shape、subject、smoothing、hemi 与数据一致。把缺失/错误精确列出。
- [ ] 对 ctxreg 与 ctxdrop 的 FC stage 同样检查实际输入输出路径；区分“历史来源记录”和“当前必须存在的运行依赖”。尤其补查尚未完成的 ctxreg FC manifests。
- [ ] 若仅 metadata 输出路径失效，先证明新旧路径指向的预期产品身份和契约相同，再提出显式、限定字段、带审计记录的迁移；不得无证据批量字符串替换或静默 fallback。迁移需符合仓库审批规则。
- [ ] 若发现真实数值/配置变化，标记依赖产品失效，给出受影响的精确产品范围及重算命令，取得必要批准后仅重算该依赖链。不得把原图存在当成来源仍有效。
- [ ] 不因个别 residual 曾经 maxdiff=0 就宣布 604 个单元完全一致；当前完整验证做了什么、没做什么分别记录。

**验收：** 活跃输入输出引用无悬空路径、无旧目录 fallback；已有重建的真实完成状态与下游有效性有证据。未通过则不宣布六图全部收尾完成。

## Task 4：同步科学说明与流程文档

**修改：** `docs/kong400_fc_contract.md`；必要时同步 `docs/kong400_fixed_cluster_reuse.md` 与实际含有错误默认路径的 launcher/docstring。

- [ ] 写明用户选择的 ctxdrop 展示定义，原始 FC 与控制后空间 cluster 的区别，以及 normalized count 的来源。
- [ ] 保留 ctxreg 回归方向与 cluster-mean 统计定义，不把 ctxdrop 改成 ctxreg，不把 0.13/0.04 设为数值正确性指标。
- [ ] Lynch 依据优先使用现有解析文件 `refs/Lynch_2024_Frontostriatal_Salience_Expansion/Lynch_2024_Frontostriatal_Salience_Expansion.md`；只引用实际读到的段落。不得声称文章有相同海马 cluster DMN Fisher-z 对照。

- [ ] 说明此次哪些是标签修改、哪些是 provenance 修复、哪些真正重新计算；更新完成声明时只写已完成且有证据的阶段。
- [ ] 若行为/路径代码发生变化，同一变更更新对应文档；环境栈未变化则不无故修改环境清单。

## Task 5：测试、渲染与六图验收

- [ ] 先确认以下测试文件存在，再运行；若仓库结构改变，查明对应测试，不悄悄跳过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/core_workflow/test_group_overview_affinity_provenance.py tests/core_workflow/test_kong400_subject_completeness.py tests/core_workflow/test_routea_ctxreg_cortex_regression.py
```

- [ ] 若 Task 3 修改了 FC/lineage 契约，另跑对应 FC/lineage 测试；报告真实命令、通过数量、warnings 和失败，不引用之前 36 passed 代替新验证。
- [ ] 确认上游稳定后，读取 `06_01_build_group_overview.py --help` 与两个现有 overview manifest，按其真实参数复用 formal group root、coverage、scene、gradient components。不得根据默认值重新选择科学参数。
- [ ] 两个 route 顺序渲染，各选现有 gradient-spectral 两个 assignment variant 及 wta-top normalized_top20_count；如生成器自动更新整份 manifest，可重绘六张，但不重算科学结果。绘图环境设单线程。

每个 route 的正式目录为 `outputs_hcp/present/overviews/<route>/`，预期三张：

```text
kong2022_gradient-spectral_group_overview_g1g2g3_<route>_fisherz.png
kong2022_gradient-spectral_group_overview_g1g2g3_<route>_normalized_top20_count.png
kong2022_wta-top_group_overview_<route>_normalized_top20_count.png
```

- [ ] 检查各 overview manifest 恰好对应三项有效产品、正确路由和来源，无另一个 sparse/dated 版本被当成正式入口；发现额外旧文件只列清单，不擅自删。
- [ ] 再验 2 routes × 2 methods × 2 smoothing × 2 hemi 的 16 个 group manifest：status、N151、有效支持域、profile 有限值；gradient 400 ROI 特征、顺序、affinity、K 与已接受配置吻合；gradient cluster profile 为 3×7。WTA 按其自身网络输出契约检查，不强行要求 3×7。
- [ ] 对比 Task 1 的数值文件哈希与科学指标：纯展示改动不得改变 cluster labels、FC、ARI、网络占比。若 Task 3 经批准重算导致改变，单独列变化和原因，不掩盖。
- [ ] 实际打开六张 PNG 检查：两张 ctxdrop gradient 说明清晰可见，count/FC 单位正确；标题/legend/colorbar 不裁切、重叠；热图负数显示与颜色一致；左右及 2mm/4mm 标签正确。
- [ ] 运行 `git diff --check`，区分本次引入的问题与已有无关问题。

## Task 6：最终交付与 Git

- [ ] 将验证命令、结果、六图路径、路径审计结果及未完成项写入已有相关文档的收尾记录，避免只有聊天记录。
- [ ] 再次检查 `git status --short` 与逐文件 diff，说明哪些修改属于此次收尾，哪些是现场已有/其他 agent 的修改。
- [ ] 不执行 `git add .`，不替其他任务提交，不 reset/checkout 覆盖工作区。若明确获准提交，仅 stage 本任务文件/对应 hunks；否则交付可审阅 diff，并明确“未提交”。不推送、不建 PR，除非用户另有授权。
- [ ] 最终向用户给出：完成内容、六张正式图链接、测试结果、运行任务已结束与否、Git 状态、剩余阻塞；不要再笼统说“全部正确”而省略 provenance 或图像检查。

## 完成门槛

- [ ] 用户选定的 ctxdrop 原始 FC 注释同时落实于两张 PNG、overview manifest 和 README/契约文档。
- [ ] 无新增统计方法，无为了数值变大改参数，无对标准分支或归档的越界操作。
- [ ] 运行中上游任务有明确结局；所有当前依赖路径与数据身份通过核验，或明确报告阻塞而不宣称完成。
- [ ] 六图及 16 组数据契约均有最新验证；纯展示改动无科学数值变化。
- [ ] 测试与 diff 检查有实际结果，交付 Git 状态清楚，未触碰未授权删除。

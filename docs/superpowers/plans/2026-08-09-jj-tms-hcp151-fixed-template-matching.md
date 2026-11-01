# JJ_TMS × HCP151 Fixed-Template Matching 执行计划

## Summary

- 先构建一次可复用的 HCP151 功能模板，再由 JJ_TMS adapter 引用，不复制模板到 JJ_TMS。
- 执行顺序：实现与测试 → 3 人 smoke（1001/1002/1003）→ 严格验证 → 固定 70 人全队列。
- 使用全部 `task-rest` XCP-D denoised runs；sub-1016 的 2-run 情况作为合法输入，其余 69 人为 3 runs。
- 计划在执行开始时保存到 `docs/superpowers/plans/2026-08-09-jj-tms-hcp151-fixed-template-matching.md`，并采用 inline `executing-plans` 工作流。

## Implementation Changes

- 抽取共享 template-matching 模块，提供两个稳定接口：
  - `build_cluster_prototypes(group_fc, labels, support, k=3) -> float32[k,400]`
  - `match_vertices_to_prototypes(fisherz_fc, valid_mask, prototypes) -> labels, scores, margin`
  - 现有 HCP cross-fit 改为调用该模块，结果与当前实现保持一致。
- 新增 HCP 模板 builder，将四种 `2mm/4mm × L/R` 的 `group_labels_final.npy`、`group_support_mask.npy` 和 `group_mean_fisherz_fc_full.npy` 转换为严格的 `3×400` prototype；不得使用聚合定义不同的 `cluster_roi_profile.npy`。
- 创建已确认的浅层共享目录：
  - `outputs_hcp/templates/hcp151_routea_full_k3_fisherz/`
  - 文件平铺命名为 `hcp151_<smoothing>_hemi-<L|R>_{labels,support,prototypes}.npy`，并写入包含 151 人、den-512、K=3、源路径和 SHA-256 的 `template_manifest.json`。
- 新增 JJ_TMS adapter：
  - 按数值 run 顺序逐 run 将 MNI XCP-D volume 采样到个体 corobl den-512 表面，保留每个 run 的 GIFTI，再沿时间轴拼接；不生成重复的大型 concat NIfTI。
  - 校验 volume 网格/affine、拼接时间点与 91k concat dtseries 完全一致。
  - 使用个体 ArealMSHBM 标签提取固定顺序的 Kong2022 400 ROI 时序；缺失 ROI 保留为 NaN，不改变列顺序。
  - 采用既有定义：海马 tSNR `10000/std ≥ 25`、2/4 mm surface smoothing、vertex-to-ROI Pearson FC、Fisher-z、与 HCP prototype 的 jointly-finite Pearson matching。
  - 输出 AP 标签 `1=Anterior/VAN, 2=Middle/SMN, 3=Posterior/DMN`、三个相关系数、top1-top2 margin、GIFTI、输入/输出指纹和 consumed-surface density evidence。
- 创建已确认的 JJ_TMS 目录：
  - `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/`
  - `outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/runs/`
- CLI 固定为 `--mode preflight|run|validate`，支持 `--subjects`、固定 roster manifest、`--run-id`、`--jobs 2`、`--resume` 和路径覆盖；不提供 `--overwrite`。失败的 `.partial` 和每-run中间产物保留，不执行删除。
- 新增方法文档和 individual-workflow README 入口；不修改当前未提交的 JJ_TMS HippUnfold 文件。没有依赖或运行时栈变化，因此无需改环境 manifest。

## Tests and Acceptance

- TDD 覆盖：
  - prototype 只使用 support 内对应标签顶点，并正确处理 NaN/Inf。
  - fixed-template 输出严格为四套 `labels(512,)`、`support(512,)`、`prototypes(3,400)`。
  - 模板 builder 拒绝非 151 人、非 den-512、非 K=3、错误 smoothing/hemi 或指纹漂移。
  - 2-run/3-run 输入均可处理；时间点、grid/affine、32k 标签长度或 ROI 顺序不匹配时立即失败。
  - 固定 400 ROI 缺失列为 NaN；matching 仅使用 jointly-finite features。
  - resume 只复用指纹、shape、dtype 和参数完全一致的产物；不覆盖不一致结果。
  - 现有 cross-fit 测试继续通过，证明共享模块抽取未改变 HCP 结果。
- 使用已确认且当前不存在的 `/tmp/jj_tms_hcp151_pytest_20260809/` 作为 pytest `--basetemp`，测试后保留，不清理。
- 每次 Python 仓库任务先记录解释器，并只使用 `/home/yojiao/.conda/envs/py314/bin/python`。
- Smoke 进入全队列的硬门槛：
  - 3/3 被试、4/4 smoothing×hemi 组合均为 `status=ok`。
  - volume 与 dtseries 时间点相等，表面为实际 consumed den-512 资产。
  - labels/scores/margin shape、dtype、范围正确；每个 valid vertex 获得非零标签，三个标签均出现。
  - 400 ROI registry 与 HCP 模板指纹一致，无输入或产物指纹漂移。
- 全队列完成标准：固定 roster 70/70、每人四组合全部通过，并生成 `status=complete` 的 full-run manifest；任何失败均保持 `in_progress/failed`，不得报告完成。

## Launch Sequence

1. 运行共享模块、新 builder、JJ adapter 和既有 cross-fit 测试。
2. 对 HCP builder 执行只读 preflight，然后生成共享模板并 validate。
3. 对 1001/1002/1003 执行 JJ_TMS preflight。
4. 以 `--jobs 2 --run-id jj_tms_hcp151_smoke_20260809` 运行 smoke，并执行 `--mode validate`。
5. 只有 smoke 硬门槛全部通过后，以固定 manifest、`--jobs 2 --resume --run-id jj_tms_hcp151_full_20260809` 运行 70 人。
6. 执行全队列 validate，核对 70/70 manifest、SHA-256、label counts、margin summaries 和 density evidence。

## Assumptions

- 仅使用 `task-rest`，不包含 TMS task runs。
- 共享模板是跨数据集只读资产；下游只记录其路径和指纹。
- 当前只生成数值、GIFTI、QC 表和 manifests，不制作 cohort overview PNG。
- 不访问 archive，不删除或覆盖任何文件；若未来需要清理 partial/test 文件，必须另行列出精确目标并取得确认。

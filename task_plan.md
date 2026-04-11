# HCP 7T HippoMaps Task Plan

## Goal
为单个 HCP 7T 被试打通当前可执行的 hippocampal functional parcellation network workflow，并在现有合法输入模型下实现 run-aware `K` instability smoke，生成协议要求的 `per-K` 记录、最终选 `K` 记录与 overview 图。

## Phases
| Phase | Status | Notes |
|---|---|---|
| 1. 初始化项目骨架与文档 | complete | 已创建文档、脚本、目录结构 |
| 2. 验证远端连通与数据发现 | complete | 已连通远端、锁定 100610、完成最小数据拷贝 |
| 3. 准备环境与官方资源 | complete | hippo 环境、Workbench 与 Schaefer atlas 已就绪 |
| 4. 实现分析脚本 | in_progress | 主线回到当前合法输入模型；run-aware instability 已改成优先使用显式 `run-1..4`，缺失时从 `run-concat` 自动拆出 4 个 run staging 后再计算 |
| 5. 运行验证与记录结果 | in_progress | `100610 + lynch2024` 已完成 `network-gradient` 与 `network-prob-cluster-nonneg` 本地 smoke；下一步收敛正式 `V_min/B`，并决定 `3 vs 5 gradients` 的正式口径 |
| 6. 修复 102311 HippUnfold surface-volume 对齐失败 | complete | 已在 `scripts/wb_command` 增加零映射检测与负-x 校正 fallback，`102311` 已于 2026-03-26 10:39 完成 145/145 steps |

## Fixed Decisions
- 工具基线：当前支线统一使用 HippUnfold CLI 2.0.0
- 海马 surface density：2mm
- 单被试优先
- 新皮层参考默认走 archive 中的 CIFTI/dtseries + Schaefer400 surface atlas
- archive 已确认存在聚合 `rfMRI_REST_7T_Atlas_1.6mm_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii`
- 海马顶点时序仍通过体空间 BOLD -> hippocampal surface 采样获得，暂不改成直接从 CIFTI 生成
- 已停止继续尝试“完全不用单独 volume 数据”；当前 K selection 实现基于现有合法海马 surface 输入推进
- 碰到环境不确定性必须先向用户汇报

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| 远端 ssh 到 192.168.0.113 报 `No route to host` | 1 | 作为环境阻塞点记录，后续继续验证但不假设可连通 |
| `rsync` 无法处理带空格的远端路径 | 1 | 改为统一使用 `ssh + binary stream` 复制远端文件 |
| 系统 `python3` 缺少 `tomllib` | 1 | 改为脚本内置 TOML 兼容解析 |
| 当前被试仅发现 volume rsfMRI，无 CIFTI | 1 | 由主控脚本显式阻断，等待用户批准 volume-based 分支 |
| 早期 conda 包名与 CLI 版本不一致 | 1 | 当前支线已统一到 `hippunfold --version = 2.0.0` |
| `hippomaps` 顶层导入触发 Qt/VTK crash | 1 | 改为项目内脚本直接按文件路径加载 `hippomaps/utils.py` |
| 直接调用 `wb_command` 在本机触发 processor/Qt 问题 | 1 | 从既有成功目录确认需用 `arch -x86_64` 调用，现已恢复可用 |
| `nilearn 0.10.2` 不支持 `copy_header` | 1 | 修正参考提取脚本以兼容当前 nilearn 版本 |
| Snakemake/HippUnfold 默认缓存写入 `~/Library/Caches` 被沙箱拒绝 | 1 | 使用 `--runtime-source-cache-path /tmp/...` 与 `HIPPUNFOLD_CACHE_DIR=/tmp/...` |
| HippUnfold 运行时缺少 `wget` | 1 | 已安装 `wget` 到 `hippo` 环境 |
| HippUnfold 模型下载受网络沙箱影响 | 1 | 改为单独用已授权网络下载模型到 `/tmp/hippunfold_cache/model` |
| `102311` 右海马 `postproc_boundary_vertices` 报 `Label 0 has less than minimum number of vertices` | 1 | 已定位为 Workbench 将全零 `sdt.shape.gii` 传给后处理，原因是 corobl surface 与 volume 在 x 轴负向 affine 下发生整体平移错位，准备在本地 `wb_command` 包装器中对零映射结果做自动负-x 校正采样 |
| 早期远端发现只查了未归档目录，误以为没有 7T CIFTI | 1 | 已在 `Resting State fMRI 7T Preprocessed Recommended archive/*.zip` 中确认聚合与分 run `dtseries` 均存在，主流程改回 surface-first |
| 严格“完全不用单独 volume 数据”要求与当前 overview workflow 冲突 | 1 | 已确认 `dtseries` 中海马为 volume grayordinates，现有 overview workflow 又要求海马 surface 输入，因此该路线停止 |

# HCP 7T HippoMaps Task Plan

## Goal
为单个 HCP 7T 被试搭建论文兼容的 HippUnfold + HippoMaps 流程，切换到 archive 中可用的 surface/CIFTI 功能输入，生成海马结构分区、surface-based 皮层参考、海马 FC 梯度与正式出图，并同步产出中文项目文档。

## Phases
| Phase | Status | Notes |
|---|---|---|
| 1. 初始化项目骨架与文档 | complete | 已创建文档、脚本、目录结构 |
| 2. 验证远端连通与数据发现 | complete | 已连通远端、锁定 100610、完成最小数据拷贝 |
| 3. 准备环境与官方资源 | complete | hippo 环境、Workbench 与 Schaefer atlas 已就绪 |
| 4. 实现分析脚本 | in_progress | 正从 volume-first 切换到 surface-first：补 archive dtseries 发现/拷贝与 CIFTI Schaefer 参考提取 |
| 5. 运行验证与记录结果 | pending | 先做脚本级验证，再抽样验证真实被试 |
| 6. 修复 102311 HippUnfold surface-volume 对齐失败 | complete | 已在 `scripts/wb_command` 增加零映射检测与负-x 校正 fallback，`102311` 已于 2026-03-26 10:39 完成 145/145 steps |

## Fixed Decisions
- 工具基线：当前机器实际可运行的 HippUnfold CLI 1.5.2-pre.2
- 海马 surface density：2mm
- 单被试优先
- 新皮层参考默认走 archive 中的 CIFTI/dtseries + Schaefer400 surface atlas
- archive 已确认存在聚合 `rfMRI_REST_7T_Atlas_1.6mm_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii`
- 海马顶点时序仍通过体空间 BOLD -> hippocampal surface 采样获得，暂不改成直接从 CIFTI 生成
- 碰到环境不确定性必须先向用户汇报

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| 远端 ssh 到 192.168.0.113 报 `No route to host` | 1 | 作为环境阻塞点记录，后续继续验证但不假设可连通 |
| `rsync` 无法处理带空格的远端路径 | 1 | 改为统一使用 `ssh + binary stream` 复制远端文件 |
| 系统 `python3` 缺少 `tomllib` | 1 | 改为脚本内置 TOML 兼容解析 |
| 当前被试仅发现 volume rsfMRI，无 CIFTI | 1 | 由主控脚本显式阻断，等待用户批准 volume-based 分支 |
| conda 包名与实际 CLI 版本不一致 | 1 | 以实际 `hippunfold --version = 1.5.2-pre.2` 为准，密度回退到 2mm |
| `hippomaps` 顶层导入触发 Qt/VTK crash | 1 | 改为项目内脚本直接按文件路径加载 `hippomaps/utils.py` |
| 直接调用 `wb_command` 在本机触发 processor/Qt 问题 | 1 | 从既有成功目录确认需用 `arch -x86_64` 调用，现已恢复可用 |
| `nilearn 0.10.2` 不支持 `copy_header` | 1 | 修正参考提取脚本以兼容当前 nilearn 版本 |
| Snakemake/HippUnfold 默认缓存写入 `~/Library/Caches` 被沙箱拒绝 | 1 | 使用 `--runtime-source-cache-path /tmp/...` 与 `HIPPUNFOLD_CACHE_DIR=/tmp/...` |
| HippUnfold 运行时缺少 `wget` | 1 | 已安装 `wget` 到 `hippo` 环境 |
| HippUnfold 模型下载受网络沙箱影响 | 1 | 改为单独用已授权网络下载模型到 `/tmp/hippunfold_cache/model` |
| `102311` 右海马 `postproc_boundary_vertices` 报 `Label 0 has less than minimum number of vertices` | 1 | 已定位为 Workbench 将全零 `sdt.shape.gii` 传给后处理，原因是 corobl surface 与 volume 在 x 轴负向 affine 下发生整体平移错位，准备在本地 `wb_command` 包装器中对零映射结果做自动负-x 校正采样 |
| 早期远端发现只查了未归档目录，误以为没有 7T CIFTI | 1 | 已在 `Resting State fMRI 7T Preprocessed Recommended archive/*.zip` 中确认聚合与分 run `dtseries` 均存在，主流程改回 surface-first |

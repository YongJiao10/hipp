# Kember 海马模板匹配核查

核查日期：2026-09-21；IDAS。定向阅读论文 HTML、作者公开源码、本地源码及指定产物；未访问 archive、未删除文件、未运行生产流程。遵循 research 技能的一手来源核查与单文件报告方式。

**结论：可以说“采用相同的逐顶点相关最大化模板指派规则”，不能说“完整复现 Kember 流程”或“数值等价”。本地明确在 Fisher-z FC 空间进行 Pearson 匹配；作者当前公开海马代码对个体 FC 未作 Fisher-z。**

## 论文与作者源码

- [Fig. 1 图注第 6 步](https://www.nature.com/articles/s41467-026-74572-1/figures/1)与 [Methods：Precision mapping of hippocampal functional systems](https://www.nature.com/articles/s41467-026-74572-1#Sec7)描述：每个海马顶点的皮层 FC profile 与三个群体系统平均 FC 模板分别相关，取最大相关对应系统。正文该段没有明确写 Pearson 或 Fisher-z；不能用后文其他分析中的 Pearson 或 z-score 反推这一步。
- 作者 [hipp_mapping/mapping.py 第 60–66 行](https://github.com/JonahKember/precision_mapping/blob/main/hipp_mapping/mapping.py#L60-L66)明确用 `np.corrcoef`，函数注释亦明确 Pearson。第 106–110 行对每个顶点与每个模板调用该函数，再 `np.argmax`。所以 Pearson 的依据来自公开实现，而非对论文“correlated”的猜测。
- [同文件第 86–110 行](https://github.com/JonahKember/precision_mapping/blob/main/hipp_mapping/mapping.py#L86-L110)：先对 BOLD 时间序列 `zscore`，计算矩阵乘积除以 `n_TRs-1`，直接进入 profile 匹配；这里没有 `arctanh` / Fisher-z。时间序列标准化不是 Fisher-z。细节：SciPy 默认总体标准差配合 `T-1` 分母，使该 FC 相对标准 Pearson 多一个统一正比例因子；该因子不影响随后 profile Pearson 相关。
- 模板直接从 `hipp_mapping.data/template.{hemi}.shape.gii` 载入。这次未核实其生成代码及 FC 尺度，也未确认当前 `main` 与论文实际运行版本完全一致。因此只能确认“所读公开个体匹配路径未作 Fisher-z”，不能断言论文所有阶段都没有 Fisher-z。
- [同文件第 115–134 行](https://github.com/JonahKember/precision_mapping/blob/main/hipp_mapping/mapping.py#L115-L134)还有小簇过滤与空间膨胀；公开程序最终标签不一定等于未经后处理的 argmax。论文另述皮层 Yeo 网络的阈值化 Dice 匹配，不能与本次海马 Pearson 匹配混用。

## 指定 Stanford 产物的实际计算证据

以下本地行号均基于本次工作区；链接指向具体文件起始证据行。

1. [subject_manifest.json:502](/home/yojiao/hpchome/hipp/outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-1016/subject_manifest.json:502)记录输入为 `shared_inputs/routea/sub-1016/fc/kong2022/2mm/hemi_L/hipp_vertex_to_network_fc.npy`，shape 为 512×400；虽然文件名含 network，实际是 400 个 Kong2022 ROI 特征。[同文件:920](/home/yojiao/hpchome/hipp/outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-1016/subject_manifest.json:920)明确记录：预计算 Pearson FC → clip ±0.999999 后 arctanh → jointly-finite Pearson 与固定 HCP151 prototypes 匹配。
2. [combo_manifest.json](/home/yojiao/hpchome/hipp/outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-1016/matching/2mm/hemi-L/combo_manifest.json:1)的 `density_evidence` 来自已消费 `den-512` surface 和 GIFTI 点数；512 并非仅由命令参数推断。该文件明确指向 `outputs_hcp/templates/hcp151_routea_full_k3_fisherz/template_manifest.json` 及 `hcp151_2mm_hemi-L_prototypes.npy`，模板 shape 为 3×400，得分 shape 为 512×3。
3. [template_matching.py:42](/home/yojiao/hpchome/hipp/scripts/core_workflow/lib/template_matching.py:42)在双方共同有限的 ROI 上去均值，再以点积除以范数乘积计算 Pearson；[第 84–99 行](/home/yojiao/hpchome/hipp/scripts/core_workflow/lib/template_matching.py:84)取最大有限相关，标签为模板索引 +1，并保存第一、第二名差值。无可用相关时标签保留 0；与作者只排除 NaN 的行为不完全相同。
4. 只读数值核对：使用 py314、禁止写字节码，读取该 combo 明确列出的数组。保存的 Fisher-z FC 与 `arctanh(clip(saved_pearson_fc, ±0.999999))` 最大绝对误差 **0.0**；把保存的 Fisher-z FC、valid mask、指定 HCP151 prototypes 传给现有 `match_vertices_to_prototypes` 后，512 个标签全部相同，保存相关得分最大绝对误差 **0.0**。这证明指定产物符合所述计算链，并不证明论文结果相等。

**调用者追踪限制：** 对当前非 archive 的 `scripts/` 做普通及包含 ignored 文件的定向检索，未找到写出 `stanford_hcp151_fixed_template_matching` 的 Stanford 历史入口；所读 run manifest 亦未检出脚本/命令引用。因此此处凭 subject manifest 和精确数值复算确认实际计算语义，不能虚构一个已找到的历史 caller 文件/行号，也不能把 JJ_TMS 或 HCP crossfit 调用者冒充 Stanford 调用者。当前范围内，历史入口身份仍未确认。

## HCP151 模板来源与差异

- [固定模板构建器:398](/home/yojiao/hpchome/hipp/scripts/individual_workflow/03_01_build_hcp151_fixed_template.py:398)第 415–418 行载入群体标签、support 和 group FC，调用 `build_cluster_prototypes`；[模板均值函数:6](/home/yojiao/hpchome/hipp/scripts/core_workflow/lib/template_matching.py:6)按簇对 supported 顶点逐 ROI 求有限均值。
- [模板 manifest:52](/home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp151_routea_full_k3_fisherz/template_manifest.json:52)明确列出 HCP 的 `group_mean_fisherz_fc_full.npy`、`group_labels_final.npy`、support 和 ROI registry；[第 314–327 行](/home/yojiao/hpchome/hipp/outputs_hcp/templates/hcp151_routea_full_k3_fisherz/template_manifest.json:314)记录 subject-mean Fisher-z FC、151 人及均值定义，随后列出 HCP subject IDs。[上游 gradient manifest:16](/home/yojiao/hpchome/hipp/outputs_hcp/work/hippomaps_gradients/routea/2mm/hemi_L/hippomaps_gradient_manifest.json:16)声明先对每人 FC 做 clipped arctanh，再跨人平均。
- 因而这是预先由 HCP151 构建、应用到 Stanford 个体的外部固定模板；所查构建链不使用 Stanford 组均值重新训练。这里的“独立”指模板构建数据源与 Stanford 应用数据源分开，不是已完成跨库受试者身份审计。
- 论文使用 7262×32492 顶点 FC、前 100 个 diffusion gradients、GMM 三类；本地使用 512×400 ROI FC，[群体 atlas manifest:382](/home/yojiao/hpchome/hipp/outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/group_atlas_manifest.json:382)明确为 spectral clustering、梯度 1/2/3、K=3、coverage 0.8。这些是特征空间、模板来源和建模算法的实质差异。
- Fisher-z 是非线性变换，通常不能保证 `corr(r, template)` 与 `corr(arctanh(r), transformed_template)` 得分或排名一致。共同使用 Pearson + argmax 不消除这一差别。

建议图示表述：**“个体海马顶点的 400-ROI Fisher-z FC profile，与独立 HCP151 三个固定群体均值模板计算 Pearson 相关，按最大相关指派系统；指派原则参照 Kember，特征与模板构建方案不同。”** 本地 Anterior/VAN、Body/SMN、Posterior/DMN 是本地标签语义，不应当作论文相同网络功能结论。

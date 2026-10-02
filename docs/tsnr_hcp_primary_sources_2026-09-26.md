# HCP tSNR 质控：一手资料核对

核对日期：2026-09-26。范围仅为文献与官方 HTML 文档；未读取影像、执行筛查或修改生产代码。按当前用户指示，HCP task151 完成前，所有新筛查继续暂停。

## 判断

本次核对的资料支持把 tSNR 用于描述信号质量、定位信号损失，并强调计算阶段和空间位置的影响；**不足以把“统一 tSNR=25、每个 run 均通过的交集、海马不合格顶点比例 ≥20% 即排除被试”称为 HCP 官方标准或由这些论文验证的组合规则**。这是对下面已核对材料的证据范围判断，并非声称所有 HCP 文献均不存在此类阈值，也不是证明 25 在本项目必然不适用。

## 一手证据

### Barch et al., 2013：任务态 tSNR 的分母具有明确模型含义

“Quality Assurance Metrics”说明：在总计 4 mm FWHM 平滑的数据上逐 run 计算 tSNR；分母来自 FEAT `sigmasquareds` 的平方根，即白化及模型拟合后的残差标准差。随后按任务跨 run、跨被试平均地图，用来显示信号损失。结果部分说明该文 20 位被试没有 run 被排除。这里的跨 run 平均不能作为逐 run 达标交集的证据，模型残差 tSNR 也不能直接等同于任意去噪时序的 mean/SD。[论文 HTML：方法与质控结果](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011498/)

### Glasser et al., 2013：皮层映射使用局部 CV 排除

“fMRISurface”步骤在体积到皮层表面映射时排除局部变异系数 CV 偏高的体素：超过 5 mm sigma 高斯邻域平均 CV 的 0.5 个标准差。作者解释采用相对局部偏离，是因为线圈敏感度和 EPI 信号脱落使 SNR 随脑区改变。这是皮层 ribbon 映射的体素选择，不能直接移植为 HippUnfold 海马顶点或被试排除规则。[论文 HTML：fMRISurface / Figure 21 附近](https://pmc.ncbi.nlm.nih.gov/articles/PMC3720813/)

### Marcus et al., 2013：快速 QC 的统计分布，不是 25 的合格线

`FUNCTIONAL_QC` 在 DICOM 转换后运行，以运动校正、脑提取和强度归一化计算逐体素 mean/SD；它比正式最小预处理流程更简单。Table 1 汇总 126 人、2207 个任务态及静息态 run：全脑掩膜内平均 tSNR 的总体均值为 24.41，标准差 2.39，第 5/95 百分位为 20.37/28.04。这是该阶段全脑汇总量的描述，既不是海马顶点分布，也不能由均值反推 25 的剔除标准。[论文 HTML：FUNCTIONAL_QC 与 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3845379/)

读取限制：Marcus 的直接打开曾返回浏览器验证页；上述信息由该一手 PMC 页的可检索正文及表格取得，未据此宣称完成全文的排除标准穷尽检索。

### 官方 HCP 2025 发布说明：处理版本必须区分

2025-08-11 发布说明列出 SEBASED 强度偏场校正、取消清理步骤中的运动回归、3T 任务态 multi-run FIX，以及所有 3T/7T fMRI 的 Reclean 和 temporal ICA；明确要求不要混用 2025 与 S1200 (2017) 数据。因处理阶段改变会改变时序统计量，不能仅凭“同为 HCP”就把旧文 tSNR 数值直接移用到新处理产物。[官方发布更新与已知问题](https://wiki.humanconnectome.org/docs/HCP%20Data%20Release%20Updates%20Known%20Issues%20and%20Planned%20fixes.html)

补充文件语境：HCP-Users 上 Glasser 对 `rfMRI_REST_hp2000_clean_rclean_tclean.nii.gz` 的答复确认，新数据具有改进的偏场校正、空间 ICA 清理和时间 ICA 清理，体积与 CIFTI 均应用这些校正。这支持辨别处理阶段；不能单凭后缀确定本项目实际消费的文件。[2026-08-06 维护者答复](https://groups.google.com/a/humanconnectome.org/g/hcp-users/c/X44pAWZ0gys)

## 对现有拟议规则的含义

- **25**：目前缺少与本项目海马输入、处理阶段和 tSNR 定义匹配的一手验证依据；保留为待论证候选值，不标记为 HCP 通用标准。
- **所有 run 取交集**：上述论文中的跨 run 平均不验证此规则。逻辑上，交集要求每个 run 都通过；增加 run 时交集不会扩大，因而 run 数和 run 集合必须明确。这是数学性质，不是 HCP 推荐。
- **≥20% 海马顶点即排除被试**：上述来源未提供该海马专属比例规则的直接依据。顶点失败、run 失败与被试失败是不同层级，不能由皮层映射的局部 CV 阈值自动推导。

后续讨论首先需要明确已消费的确切文件和版本、分母为时序 SD 还是模型残差 SD、空间平滑及海马采样方式；本次不执行这些数据核查，也不启动任何新筛查。

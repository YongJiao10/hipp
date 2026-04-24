# MSHBM Simple Flowchart Notes

配套图文件：`docs/hipp_parcellation_network/mshbm_simple_flowchart.drawio`

## Purpose
这张图用于汇报场景，目标是把 MSHBM 的核心逻辑压缩成“能讲清、能记住”的最小链路。

## Main Flow (Talk Track)
1. Input:
海马顶点时序 + 参考网络标签，构建特征矩阵 `X(v,d)`。

2. Step 1 Initialization:
做跨被试 ROI 交集与球面 k-means，得到初始 cluster/lambda。

3. Step 2 Group EM:
E-step 更新 `lambda`，M-step 更新 `mu / sigma / epsil / kappa / theta`。

4. Key Group Product:
`theta(v,n)` 是最终 group priors（顶点到网络的先验概率分布）。

5. Tune:
在 validation 上搜索 `(w,c)`：
- `w`: group prior 强度
- `c`: 空间平滑强度

6. Step 3 Individual Inference:
`score = data term + w*log(theta) - c*smoothness`
输出个体 posterior 与 labels。

## Variable Interpretation (for Audience)
- `X(v,d)`: 顶点-特征矩阵（图里用网格矩阵表示）
- `mu(:,n)`: 第 `n` 个网络的模板向量（图里用列向量矩阵表示）
- `theta(v,n)`: group prior，最值得展示和解释的核心结果

## Recommended Slide Usage
- 主讲只讲主链路（Input -> Step1 -> Step2 -> Tune -> Step3 -> Results）
- 变量解释只保留三句：
  - `theta` 是最终 group prior
  - `w` 控制“贴近组先验”
  - `c` 控制“空间连续性”

## Open/Edit
可在 draw.io / diagrams.net 直接打开 `.drawio` 文件继续调整配色和文案。

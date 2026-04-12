# Hippocampal K Selection by Instability: Protocol and Recording Template

## 1) Goal

本文件用于固定并记录海马功能分区中基于 `instability` 曲线的 `K` 选择流程，目标是：

1. `data-driven` 选 `K`，但控制分析自由度。
2. 避免“`K` 越小越稳定”的天然偏置直接主导结论。
3. 保证每一步都有可追溯记录，便于审稿核查。

## 2) Inputs

输入至少包括：

1. 海马顶点级功能特征（例如 `vertex-to-network FC` 或其变换后特征）。
2. 可独立重采样的数据分块（优先 `session/run` 级分割）。
3. 空间邻接信息（用于空间约束聚类）。
4. 预先定义的候选 `K` 范围与随机种子列表。

## 3) Candidate K Range

### 3.1 `K_min`

固定设为 `K_min = 2`。

### 3.2 `K_max`

用“最小可解释分区大小”约束确定：

`K_max <= floor(N_vertex / V_min)`

其中：

1. `N_vertex` 是该半球海马有效顶点数。
2. `V_min` 是最小可接受分区顶点数（预先固定，不可后改）。

### 3.3 推荐起始范围

若无更强先验，可先用 `K = 2..10`，再由 `V_min` 约束裁剪无效 `K`。

## 4) Instability Curve Computation

对每个 `K` 重复 `B` 次重采样（建议 `B >= 100`）：

1. 生成两份独立子数据 `A` 与 `B`（同一被试内的独立 session/run 或时间块）。
2. 在 `A` 上做空间约束聚类得到标签 `L_A(K,b)`。
3. 在 `B` 上做同配置聚类得到标签 `L_B(K,b)`。
4. 用 Hungarian matching 对齐 `L_A` 与 `L_B` 的标签编号。
5. 计算一致性 `S(K,b)`（主指标推荐 `ARI`）。
6. 定义不稳定性 `I(K,b) = 1 - S(K,b)`。

聚合得到：

1. `I_mean(K) = mean_b I(K,b)`
2. `I_se(K) = sd_b(I) / sqrt(B)`
3. 可选 `95% CI`（bootstrap）

## 5) Anti-Small-K Bias Rules

不能直接用 `I_mean(K)` 的全局最小值选 `K`。必须同时应用以下规则：

1. 只在 `I_mean(K)` 的局部极小值集合中选候选 `K`。
2. 使用 chance-corrected 指标（例如 `ARI/AMI`）或对 `I` 做 `null` 校正。
3. 应用 `1-SE` 规则：在最优点 `1 SE` 内选最小且可解释的 `K`。
4. 应用非平凡性约束：分区内同质性、最小分区大小、空间连通性均达标。

## 6) Final K Decision

最终主分析 `K*` 按以下顺序确定：

1. 先筛出稳定性局部极小值。
2. 再做 `1-SE` 筛选。
3. 再过非平凡性约束。
4. 若仍有多个候选，选复杂度最低且外部效度不劣者。

并固定敏感性分析为 `K* - 1` 与 `K* + 1`（若在候选范围内）。

## 7) Required Records

以下表格为必填模板。建议每次正式运行都完整填写并归档。

### 7.1 Run Metadata

```text
Field                         Value
----------------------------  ---------------------------------------------------------
project_id                    <string>
analysis_date                 <YYYY-MM-DD>
operator                      <name>
code_commit                   <git hash>
feature_definition            <e.g., vertex-to-network FC + fisher-z + zscore>
hemisphere                    <L/R>
subject_set                   <subject IDs or cohort name>
split_strategy                <session split / run split / time-window split>
clustering_method             <e.g., spatial Ward>
distance_metric               <e.g., correlation / euclidean>
spatial_constraints           <adjacency file/path>
random_seed_policy            <fixed list / deterministic rule>
B_resamples                   <int>
K_min                         <int>
K_max                         <int>
V_min                         <int>
```

### 7.2 Per-K Summary

```text
K   n_valid_resamples   I_mean(1-ARI)   I_se      local_min   within_1SE_best   homogeneity   min_parcel_ok   connectivity_ok   null_corrected_score
--  ------------------  --------------  --------  ----------  -----------------  ------------  --------------  ----------------  --------------------
2   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
3   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
4   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
5   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
6   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
7   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
8   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
9   <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
10  <int>               <float>         <float>   <0/1>       <0/1>              <float>       <0/1>           <0/1>             <float>
```

### 7.3 Final Selection Log

```text
Item                          Decision
----------------------------  ---------------------------------------------------------
best_by_instability           <K>
candidate_local_minima        <[K1,K2,...]>
1SE_selected                  <K>
post_constraint_selected      <K*>
main_analysis_K               <K*>
sensitivity_K                 <K*-1, K*+1 or NA>
primary_reason                <one-sentence method reason>
secondary_reason              <one-sentence biological/practical reason>
deviations_from_protocol      <none or detailed note>
```

## 8) Reporting Checklist

主文或补充材料至少报告：

1. 候选 `K` 范围与其确定依据（`V_min` 规则）。
2. `instability` 计算方式（重采样策略、`B`、一致性指标、是否 `null` 校正）。
3. 选 `K` 规则（局部极小值 + `1-SE` + 约束）。
4. 主 `K*` 与 `K*±1` 的敏感性结果。
5. 训练/验证/测试或交叉拟合隔离策略。

## 9) Locking Rule

本协议一旦开始正式分析不得中途改动。若必须改动，需：

1. 新建版本号（例如 `v1 -> v2`）。
2. 在 `deviations_from_protocol` 中写明改动时间、原因、影响范围。
3. 旧版本与新版本并行保留，禁止覆盖。


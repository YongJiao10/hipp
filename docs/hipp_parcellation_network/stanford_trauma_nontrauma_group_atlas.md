# Stanford 创伤 (Trauma) 与 症状 (Symptom) 海马群组分区与概览文档

本文档记录 Stanford CausalConnectome 数据集按 **二分维度 (HC / Symp / Trauma / Non-Trauma)** 以及 **2×2 细分亚组 (NTHC / TEHC / NTS / TIS)** 执行 `core_workflow` 的海马群组图谱结果及路径规范。

## 1. 队列定义与被试清单

Stanford 194 名合格被试由 4 类号段组成：
* `sub-1xxx` (51人): **NTHC** (Non-Trauma Healthy Control)
  * 清单路径: `outputs_stanford/inputs/kong2022/manifests/stanford_nthc_1xxx_qc_pass_t1only_51.txt`
* `sub-2xxx` (40人): **TEHC** (Trauma-Exposed Healthy Control)
  * 清单路径: `outputs_stanford/inputs/kong2022/manifests/stanford_tehc_2xxx_qc_pass_t1only_40.txt`
* `sub-3xxx` (56人): **NTS** (Non-Trauma Symptomatic)
  * 清单路径: `outputs_stanford/inputs/kong2022/manifests/stanford_nts_3xxx_qc_pass_t1only_56.txt`
* `sub-4xxx` (47人): **TIS** (Trauma-Induced Symptomatic)
  * 清单路径: `outputs_stanford/inputs/kong2022/manifests/stanford_tis_4xxx_qc_pass_t1only_47.txt`

### 二分组合对照
* **非创伤组 (Non-Trauma, NT)**: 107 人 (`1xxx` + `3xxx`) — `outputs_stanford/inputs/kong2022/manifests/stanford_nontrauma_1xxx3xxx_qc_pass_t1only_107.txt`
* **创伤组 (Trauma, TE+TI)**: 87 人 (`2xxx` + `4xxx`) — `outputs_stanford/inputs/kong2022/manifests/stanford_trauma_2xxx4xxx_qc_pass_t1only_87.txt`
* **健康组 (Healthy Control, HC)**: 91 人 (`1xxx` + `2xxx`) — `outputs_stanford/inputs/kong2022/manifests/stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt`
* **症状组 (Symptomatic, Symp)**: 103 人 (`3xxx` + `4xxx`) — `outputs_stanford/inputs/kong2022/manifests/stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt`

### 诊断细分亚组 (Diagnostic Subgroups)
* **NTS Any MDD (`nts_any_mdd`)**: 43 人 — `outputs_stanford/inputs/kong2022/manifests/stanford_nts_any_mdd_3xxx_qc_pass_t1only_43.txt`
* **NTS Pure MDD (`nts_pure_mdd`)**: 11 人 — `outputs_stanford/inputs/kong2022/manifests/stanford_nts_pure_mdd_3xxx_qc_pass_t1only_11.txt`
* **TIS PTSD (tis_ptsd)**: 35 人 — outputs_stanford/inputs/kong2022/manifests/stanford_tis_ptsd_4xxx_current_ptsd_qc_pass_t1only_35.txt
  * 溯源：在 data/CC_RC_Master_Database_10.23.18_Data.csv 中按 CC_Post_Intake_ID 连接现有 47 人 TIS QC roster，并筛选 Current_PTSD_Diagnosis == 1；机器可读证据见 stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/reports/stanford_tis_ptsd_provenance.json。


> **TIS PTSD 说明：** 该 35 人清单是 TIS 内嵌的诊断子组，仅接入 Stanford fast-rest direct-to-NTHC workflow；本次没有为它生成新的 gradient/group-atlas 产物，也不把它作为正式 NTHC/TEHC/NTS/TIS 四组 factorial 的第五个 cell。现有 NTS Pure MDD 11 人清单仍是历史固定输入，尚未按同一 Master Database 规则重新派生。

---

## 2. 运行流程及产物路径

### 2.1 群组梯度计算 (`05_01_compute_hippomaps_gradients.py`)
* NTHC: `outputs_stanford/work/hippomaps_gradients_by_group/nthc`
* TEHC: `outputs_stanford/work/hippomaps_gradients_by_group/tehc`
* NTS: `outputs_stanford/work/hippomaps_gradients_by_group/nts`
* TIS: `outputs_stanford/work/hippomaps_gradients_by_group/tis`
* Non-Trauma: `outputs_stanford/work/hippomaps_gradients_by_group/nontrauma`
* Trauma: `outputs_stanford/work/hippomaps_gradients_by_group/trauma`
* HC: `outputs_stanford/work/hippomaps_gradients_by_group/hc`
* Symp: `outputs_stanford/work/hippomaps_gradients_by_group/symp`
* NTS Any MDD: `outputs_stanford/work/hippomaps_gradients_by_group/nts_any_mdd`
* NTS Pure MDD: `outputs_stanford/work/hippomaps_gradients_by_group/nts_pure_mdd`

### 2.2 群组图谱聚类 (`05_02_run_group_atlas.py`)
* NTHC: `outputs_stanford/work/group_atlas/routea_nthc_full_k3_fisherz`
* TEHC: `outputs_stanford/work/group_atlas/routea_tehc_full_k3_fisherz`
* NTS: `outputs_stanford/work/group_atlas/routea_nts_full_k3_fisherz`
* TIS: `outputs_stanford/work/group_atlas/routea_tis_full_k3_fisherz`
* Non-Trauma: `outputs_stanford/work/group_atlas/routea_nontrauma_full_k3_fisherz`
* Trauma: `outputs_stanford/work/group_atlas/routea_trauma_full_k3_fisherz`
* HC: `outputs_stanford/work/group_atlas/routea_hc_full_k3_fisherz`
* Symp: `outputs_stanford/work/group_atlas/routea_symp_full_k3_fisherz`
* NTS Any MDD: `outputs_stanford/work/group_atlas/routea_nts_any_mdd_full_k3_fisherz`
* NTS Pure MDD: `outputs_stanford/work/group_atlas/routea_nts_pure_mdd_full_k3_fisherz`

### 2.3 加权图谱分支 (`05_05_build_weighted_fisherz_branch.py`)
* NTHC: `outputs_stanford/work/group_atlas/routea_nthc_weighted_fisherz_full_k3`
* TEHC: `outputs_stanford/work/group_atlas/routea_tehc_weighted_fisherz_full_k3`
* NTS: `outputs_stanford/work/group_atlas/routea_nts_weighted_fisherz_full_k3`
* TIS: `outputs_stanford/work/group_atlas/routea_tis_weighted_fisherz_full_k3`
* Non-Trauma: `outputs_stanford/work/group_atlas/routea_nontrauma_weighted_fisherz_full_k3`
* Trauma: `outputs_stanford/work/group_atlas/routea_trauma_weighted_fisherz_full_k3`
* HC: `outputs_stanford/work/group_atlas/routea_hc_weighted_fisherz_full_k3`
* Symp: `outputs_stanford/work/group_atlas/routea_symp_weighted_fisherz_full_k3`
* NTS Any MDD: `outputs_stanford/work/group_atlas/routea_nts_any_mdd_weighted_fisherz_full_k3`
* NTS Pure MDD: `outputs_stanford/work/group_atlas/routea_nts_pure_mdd_weighted_fisherz_full_k3`

### 2.4 可视化概览与呈现 (`06_01_build_group_overview.py`)
* NTHC Plain / Weighted: `outputs_stanford/present/overviews/routea_nthc`
* TEHC Plain / Weighted: `outputs_stanford/present/overviews/routea_tehc`
* NTS Plain / Weighted: `outputs_stanford/present/overviews/routea_nts`
* TIS Plain / Weighted: `outputs_stanford/present/overviews/routea_tis`
* Non-Trauma Plain / Weighted: `outputs_stanford/present/overviews/routea_nontrauma`
* Trauma Plain / Weighted: `outputs_stanford/present/overviews/routea_trauma`
* HC Plain / Weighted: `outputs_stanford/present/overviews/routea_hc`
* Symp Plain / Weighted: `outputs_stanford/present/overviews/routea_symp`
* NTS Any MDD (Plain, Weighted, Normalized WTA): `outputs_stanford/present/overviews/routea_nts_any_mdd`
* NTS Pure MDD (Plain, Weighted, Normalized WTA): `outputs_stanford/present/overviews/routea_nts_pure_mdd`

---

## 3. K=3 聚类网络映射结果总结 (2mm Fisher-z)

| 细分亚组 / 大组 | 样本量 (N) | 半球 | Cluster 1 (前/下) | Cluster 2 (体/中间) | Cluster 3 (后/背) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **NTHC** (`1xxx`) | 51人 | **L** | VentralAttention ($z \approx 0.036$) | Somatomotor ($z \approx 0.067$) | Default ($z \approx 0.112$) |
| | | **R** | VentralAttention ($z \approx 0.032$) | Somatomotor ($z \approx 0.075$) | Default ($z \approx 0.099$) |
| **TEHC** (`2xxx`) | 40人 | **L** | Auditory ($z \approx 0.043$) | Somatomotor ($z \approx 0.097$) | Default ($z \approx 0.131$) |
| | | **R** | Auditory ($z \approx 0.024$) | Somatomotor ($z \approx 0.104$) | Default ($z \approx 0.114$) |
| **NTS** (`3xxx`) | 56人 | **L** | Auditory ($z \approx 0.033$) | Somatomotor ($z \approx 0.081$) | Default ($z \approx 0.139$) |
| | | **R** | Auditory ($z \approx 0.037$) | Somatomotor ($z \approx 0.089$) | Default ($z \approx 0.111$) |
| **TIS** (`4xxx`) | 47人 | **L** | Auditory ($z \approx 0.061$) | Somatomotor ($z \approx 0.086$) | Default ($z \approx 0.128$) |
| | | **R** | Somatomotor ($z \approx 0.050$) | Somatomotor ($z \approx 0.083$) | Default ($z \approx 0.117$) |
| **Non-Trauma** (`1xxx`+`3xxx`) | 107人 | **L** | VentralAttention ($z \approx 0.030$) | Somatomotor ($z \approx 0.075$) | Default ($z \approx 0.126$) |
| | | **R** | Somatomotor ($z \approx 0.032$) | Somatomotor ($z \approx 0.083$) | Default ($z \approx 0.105$) |
| **Trauma** (`2xxx`+`4xxx`) | 87人 | **L** | Auditory ($z \approx 0.053$) | Somatomotor ($z \approx 0.090$) | Default ($z \approx 0.131$) |
| | | **R** | Auditory ($z \approx 0.038$) | Somatomotor ($z \approx 0.093$) | Default ($z \approx 0.113$) |
| **NTS Any MDD** | 43人 | **L** | Auditory ($z \approx 0.040$) | Somatomotor ($z \approx 0.089$) | Default ($z \approx 0.141$) |
| | | **R** | Auditory ($z \approx 0.036$) | Somatomotor ($z \approx 0.091$) | Default ($z \approx 0.115$) |
| **NTS Pure MDD** | 11人 | **L** | Somatomotor ($z \approx 0.060$) | Somatomotor ($z \approx 0.074$) | Default ($z \approx 0.137$) |
| | | **R** | Auditory ($z \approx 0.056$) | Somatomotor ($z \approx 0.070$) | Default ($z \approx 0.110$) |

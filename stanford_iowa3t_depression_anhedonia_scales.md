# Stanford 与 Iowa 3T 数据集抑郁（Depression）与快感缺失（Anhedonia）量表汇总

本文档总结了 **Stanford (Causal Connectome)** 与 **Iowa 3T (UIowa JJ_TMS)** 两个数据集中关于抑郁症（Depression）及快感缺失（Anhedonia）的临床量表、行为测验、诊断访谈及相关指标。所有信息均基于仓库内的原始元数据表及数据字典直接验证核对。

---

## 一、数据源与被试覆盖范围

| 数据集 | 权威元数据路径 | 总样本量 | 核心影像被试队列 | ID 关联字段 |
| :--- | :--- | :---: | :--- | :--- |
| **Stanford (Causal Connectome)** | `/home/yojiao/hpchome/hipp/data/CC_RC_Master_Database_10.23.18_Data.csv` | N = 332 | 194 人（NTHC=51, TEHC=40, NTS=56, TIS=47；含嵌套临床组 NTS Pure MDD=11, TIS PTSD=35） | `CC_Post_Intake_ID`（对应 `sub-4xxx` 等） |
| **Iowa 3T (UIowa JJ_TMS)** | `/home/yojiao/hpchome/hipp/outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx`<br>以及 `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx` | N = 95 | 94 人 QC 通过（67 名健康对照 HC，27 名患者；核心诊断分组排除 sub-3003 后为 26 人：7 Pure MDD + 19 Comorbid MDD） | `record_subject_id`（对应 `JJ_TMS_xxxx` / `sub-1xxx`, `sub-3xxx`） |

---

## 二、Stanford 数据集 (Causal Connectome)

### 1. 抑郁（Depression）主要量表与指标
* **PHQ-9 (Patient Health Questionnaire)**：
  * 涵盖多个阶段的时间点追踪与均值：
    * `PHQ_Score_Screen`（初筛阶段得分）
    * `PHQ_Score_Intake`（临床入组访谈得分）
    * `PHQ_Score_mean`（初筛与入组均值）
    * `PHQ_Scan_1`（第 1 次 MRI 扫描时评分）
    * `PHQ_Scan_2`（第 2 次 MRI 扫描/TMS干预时评分）
    * `elec_phq`（电生理阶段评分）
* **BDI (Beck Depression Inventory)**：
  * 变量字段：`BDI`（有效样本量 N = 198，分值范围 0–49）。
  * 贝克抑郁自评量表经典版本，测量过去一周抑郁症状严重度。
* **MASQ (Mood and Anxiety Symptom Questionnaire，情绪与焦虑症状问卷)**：
  * `MASQ_G`（General Distress - Depressive Symptoms，一般抑郁痛苦维度，N = 200，分值 5–56）。
  * `MASQ_A`（Anxious Arousal，躯体焦虑唤醒维度，N = 198，分值 5–50）。
* **WebNeuro 计算机化情绪认知测验套件 (Test Battery)**：
  * `Depressed_Mood_Level_Norm` & `Depressed_Mood_Level_Sten`：抑郁心境常模/标准十分分（N = 194）。
  * `sDepress` & `sDepress_norm`：抑郁症状原始得分及标准化分数。
* **精神科结构化临床诊断**：
  * `Current_MDD`：当前重性抑郁障碍诊断（0 = 无，1 = 有）。
  * `Lifetime_MDD`：终身重性抑郁障碍诊断（0 = 无，1 = 有）。

### 2. 快感缺失（Anhedonia）核心与相关指标
* **`MASQ_D` (Anhedonic Depression 子量表) ——【核心量表】**：
  * 变量字段：`MASQ_D`（有效样本量 N = 198，分值范围 5–50）。
  * **临床意义**：来自 Clark & Watson 抑郁-焦虑三分模型（Tripartite Model）。该模型将负性情绪（General Distress）作为抑郁与焦虑共有成分，而**低积极情感/快感缺失（Low Positive Affect / Anhedonia）**是抑郁症特异的核心特征。`MASQ_D` 是量化快感缺失维度的标杆量表。
* **PHQ-9 内嵌核心快感缺失条目**：
  * 对应 DSM-5 诊断核心标准 A2：“*Little interest or pleasure in doing things*”（做事缺乏兴趣或乐趣）。
* **WebNeuro 情绪偏向性测试**：
  * `Positivity_Negativity_Bias_Norm`、`Emotion_Bias_Norm`：测量被试对积极与消极面孔/情绪刺激的处理偏向，直接反映奖赏敏感性缺失与消极注意偏向。
* **奖赏功能 fMRI 行为范式 —— MID 任务**：
  * **Monetary Incentive Delay (MID) 任务**：数据库中记录了练习与正式任务表现（`Practice_Score_MID`, `Task_Score_MID`, `MID_Done`）。MID 是神经影像领域公认探究奖赏预期（Reward Anticipation）与奖赏反馈/消费（Reward Feedback/Outcome）的核心范式，直接对应快感缺失的腹侧纹状体-海马奖赏神经回路机制。

---

## 三、Iowa 3T 数据集 (UIowa JJ_TMS)

Iowa 3T 数据在 `outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx` 的 `data` 与 `dictionary` 表中包含完整的条目级明细（数据完成度高达 94/95）：

### 1. 抑郁（Depression）主要量表与指标
* **PHQ-9 (Patient Health Questionnaire-9)**：
  * 总分字段：`phq_score`（Visit 1 记录）、`phq_total_score`（94 人有效）、`phq_total_score_w_suicide`。
  * 条目字段：`phq_1` 至 `phq_9`，以及自杀意念 `phq_suicide` 和持续时间 `phq_10`。
* **BDI-II (Beck Depression Inventory-II，贝克抑郁量表第 2 版)**：
  * 总分字段：`bdi_score`（Visit 1 记录）、`bdi_total`（94 人有效）、`bdi_total_w_suicide`。
  * 条目字段：完整保存 1 至 21 项（涵盖悲伤 `bdi_1`、悲观 `bdi_2`、内疚 `bdi_5`、自厌 `bdi_7`、疲倦 `bdi_20` 等）。
* **MADRS-SR (Montgomery-Åsberg Depression Rating Scale - Self-Report)**：
  * 总分字段：`madrs_score`、`madrs_total`（94 人有效）、`madrs_total_w_suicide`。
  * 涵盖 9 项临床抑郁评估（情绪低落 `madrs_1`、紧张 `madrs_2`、睡眠 `madrs_3`、食欲 `madrs_4`、注意力 `madrs_5`、行动力 `madrs_6`、悲观 `madrs_8`、对生活的热情/厌世 `madrs_9`）。
* **NIH Toolbox 情绪领域 (Emotion Domain - Sadness CAT)**：
  * `nih_sad_tscore` & `nih_sad_tscore_se`（N = 94）：计算机自适应测验（CAT）生成的负性情感/悲伤 T 评分。
* **DSM-5 1 级横切症状量表 (DSM-5 Level 1 Cross-Cutting Measures)**：
  * 总分/条目字段：`dsm5crossad_2`（近两周感到低落、抑郁或绝望的频率）。
* **M.I.N.I. 7.0.2 结构化诊断访谈 (Mini International Neuropsychiatric Interview)**：
  * Module A：全面评估 DSM-5 重性抑郁发作（MDE，包括当前、既往及复发）。
  * Module B：自杀风险评估（Suicidality）。
  * 确立被试组别：Pure MDD（无共病抑郁组，N = 7）与 Comorbid MDD（伴 GAD/PTSD 共病组，N = 19）。

### 2. 快感缺失（Anhedonia）核心与相关指标
* **PANAS-SF (Positive and Negative Affect Schedule - Short Form) ——【核心量表】**：
  * 字段：`panas_positive` / `panas_pos_score`（Visit 1 前测）、`visit2_panas_pos_score`（Visit 2）、`panas_negative`。
  * **临床意义**：积极情感（Positive Affect, PA）由热情、活跃、警惕、自豪、专注等组成。在当代精神医学研究中，**低 PA 分值（Deficit in Positive Affect）被视为快感缺失（Anhedonia）最主要的连续谱度量**。
* **BDI-II 内嵌快感缺失条目**：
  * `bdi_4`：**Loss of Pleasure**（愉悦感丧失 —— 直接评估体验性快感缺失）。
  * `bdi_12`：**Loss of Interest**（兴趣减退 —— 直接评估兴趣/动机型缺失）。
  * `bdi_21`：**Loss of Interest in Sex**（性欲与性快感缺失）。
* **MADRS-SR 内嵌快感与情感卷入条目**：
  * `madrs_7`：**Emotional Involvement**（*“assess your interest in your surroundings, in other people, and in activities that normally give you pleasure”* —— 评估对日常娱乐及人际愉悦体验的情感投入）。
  * `madrs_6`：**Initiative**（意志减退/内心阻力，动机型快感缺失相关）。
* **NIH Toolbox 心理幸福感 (Positive Affect CAT)**：
  * `nih_posaffect_tscore` & `nih_posaffect_tscore_se`（N = 94）：积极情感 T 分，直接反向指标化快感缺失。
  * `nih_genlifesat_tscore`（生活满意度）与 `nih_meaning_tscore`（生命意义感）。
* **PHQ-9 与 DSM-5 横切量表第 1 题**：
  * `phq_1` 与 `dsm5crossad_1`：*“During the past two weeks, how often have you been bothered by having little interest or pleasure in doing things?”*。

---

## 四、两数据集量表对照与跨数据集对齐建议

| 维度 | **Stanford (Causal Connectome)** | **Iowa 3T (UIowa JJ_TMS)** | 对齐/研究建议 |
| :--- | :--- | :--- | :--- |
| **标准自评抑郁量表** | • **PHQ-9**（Screen/Intake/Scan1/Scan2）<br>• **BDI**<br>• **MASQ_G**（一般抑郁痛苦） | • **PHQ-9**（Visit 1 总分及 1-9 题）<br>• **BDI-II**（Visit 1 总分及 1-21 题）<br>• **MADRS-SR**（总分及 1-9 题）<br>• **NIH Toolbox Sadness**（`nih_sad_tscore`） | **强推荐使用 PHQ-9 与 BDI** 作为两数据集联合分析的核心抑郁严重度指标。 |
| **快感缺失专门量表 (Anhedonia)** | • **MASQ_D**（Anhedonic Depression 专门量表，N=198）<br>• MID 奖赏加工任务行为数据<br>• WebNeuro 积极情绪偏差 | • **PANAS-SF Positive Affect（积极情感 PA）**<br>• **BDI-II Item 4（Loss of Pleasure）**<br>• **MADRS-SR Item 7（Emotional Involvement）**<br>• **NIH Toolbox Positive Affect**（`nih_posaffect_tscore`） | 1. 跨数据集可直接提取 **PHQ-9 Item 1** 与 **BDI/BDI-II Item 4/12**。<br>2. 若进行多尺度建模，Stanford 可使用 **MASQ_D + MID 任务**；Iowa 可使用 **PANAS-PA + BDI Item 4 + MADRS Item 7**。 |
| **诊断与亚型分群** | • SCID / 临床访谈：`Current_MDD`, `Lifetime_MDD` | • M.I.N.I. 7.0.2：区分 Pure MDD (7) 与 Comorbid MDD (19) | 对应抑郁发作诊断状态及共病分型。 |
| **共病与控制量表** | • **GAD-7**（焦虑）<br>• **PCL**（PTSD）<br>• **STAI**（状态-特质焦虑）<br>• **CAPS**（临床创伤访谈） | • **GAD-7**（焦虑）<br>• **PCL-5**（PTSD）<br>• **STAI**（状态-特质焦虑）<br>• **WHODAS 2.0**（社会功能受损） | GAD-7、PCL、STAI 三大量表在两数据集中完全对称，可作为协变量或共病严重度控制变量。 |

---

## 五、在代码中直接读取的快捷方式

```python
import pandas as pd
import openpyxl

# 1. 读取 Stanford 数据（含抑郁、快感缺失、认知与精神运动指标）
stanford_csv = "/home/yojiao/hpchome/hipp/data/CC_RC_Master_Database_10.23.18_Data.csv"
df_stanford = pd.read_csv(stanford_csv, low_memory=False)
stanford_pheno = df_stanford[[
    "CC_Post_Intake_ID", "Current_MDD", "Lifetime_MDD",
    # 抑郁与快感缺失
    "PHQ_Score_Screen", "PHQ_Score_Intake", "PHQ_Score_mean", "PHQ_Scan_1", "PHQ_Scan_2",
    "BDI", "MASQ_D", "MASQ_G", "MASQ_A",
    "Depressed_Mood_Level_Norm", "Positivity_Negativity_Bias_Norm",
    # Psychomotor (精神运动)
    "Motor_Coordination_Norm", "Processing_Speed_Norm", "tdomnk_norm", "chlrrtav_norm",
    # Rumination / Worry (反刍与重复消极思维)
    "PSWQ", "ERQ_REAP", "ERQ_SUP",
    # Memory (记忆功能)
    "Working_Memory_Norm", "Recall_Memory_Norm", "digitsp", "digitot", "ctmlearn_norm", "ctmrec4_norm",
    # Attention (注意功能)
    "Sustained_Attention_Norm", "Controlled_Attention_Norm", "g2acck_norm", "g2sdrtk_norm",
    # 智力与推理
    "WASI_Matrix_T_Score", "IQ_Score"
]]

# 2. 读取 Iowa 3T (JJ_TMS) 数据（含条目级临床量表）
iowa_xlsx = "/home/yojiao/hpchome/hipp/outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx"
wb = openpyxl.load_workbook(iowa_xlsx, data_only=True)
data_sheet = wb["data"]
rows = list(data_sheet.iter_rows(values_only=True))
df_iowa = pd.DataFrame(rows[1:], columns=rows[0])
iowa_pheno = df_iowa[[
    "record_subject_id", "enrollment_status",
    # 抑郁与快感缺失
    "phq_score", "phq_total_score", "phq_1", "phq_2",
    "bdi_score", "bdi_total", "bdi_1", "bdi_4", "bdi_12", "bdi_21",
    "madrs_score", "madrs_total", "madrs_1", "madrs_6", "madrs_7",
    "panas_pos_score", "panas_neg_score",
    "nih_sad_tscore", "nih_posaffect_tscore",
    "dsm5crossad_1", "dsm5crossad_2",
    # Psychomotor (精神运动性迟滞/激越)
    "phq_8", "bdi_11", "bdi_20", "madrs_6", "gad_5", "stai_t2",
    # Rumination / Worry (消极反刍/担忧/内疚自责)
    "dsm5crossad_16", "gad_2", "gad_3", "stai_s7", "stai_t9", "madrs_8",
    "bdi_2", "bdi_5", "bdi_7", "bdi_8",
    # Memory (主观记忆障碍与日常功能)
    "dsm5crossad_15", "whodas_3", "pcl_1", "pcl_8",
    # Attention (注意力集中与维持困难)
    "phq_7", "bdi_19", "bdi_13", "madrs_5", "whodas_6", "pcl_19", "panas_17"
]]
```

---

## 六、扩展认知与精神病理维度：Psychomotor, Rumination, Memory, Attention

除抑郁总分与快感缺失外，精神病理学与认知神经科学重点关注的精神运动（Psychomotor）、反刍思维（Rumination）、记忆（Memory）和注意（Attention）四大维度在两数据集中均有高价值的指标支撑：

### 1. Psychomotor（精神运动与动作协调）
* **Stanford (Causal Connectome) ——【客观计算机化神经认知测验】**：
  * `Motor_Coordination_Norm` / `Motor_Coordination_Sten`（N = 194）：WebNeuro 运动协调常模/十分分，主要基于手指敲击测验（Finger Tapping Task，如优势手敲击 `tdomnk_norm`），客观测量精细运动速度与运动协调性。
  * `Processing_Speed_Norm` / `Processing_Speed_Sten`（N = 197）：WebNeuro 处理速度常模/十分分，基于选择反应时测验（Choice Reaction Time，`chlrrtav_norm`），衡量中枢感觉运动整合与精神运动处理敏捷度。
* **Iowa 3T (UIowa JJ_TMS) ——【临床条目级评估】**：
  * **PHQ-9 Item 8 (`phq_8`)**（N = 95）：*“Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?”*（DSM-5 精神运动性迟滞/激越金标准条目）。
  * **BDI-II Item 11 (`bdi_11`)**（N = 94）：*Agitation*（坐立不安/激越程度）。
  * **BDI-II Item 20 (`bdi_20`)**（N = 94）：*Tiredness or Fatigue*（易疲倦与精神运动迟钝）。
  * **MADRS-SR Item 6 (`madrs_6`)**（N = 94）：*Initiative*（意志力减退/行动启动阻力，直接评估从静止状态启动日常活动的阻力）。
  * **M.I.N.I. 7.0.2 结构化访谈**：`mini_a3c`（近两周精神运动性迟缓/激越诊断评估，患者组 N = 27 有效）。
  * **焦虑相关激越条目**：GAD-7 `gad_5`（坐立不安）与 STAI `stai_t2`（紧张烦躁）。

### 2. Rumination（反刍思维、担忧 Worry 与重复消极思维）
* **Stanford (Causal Connectome) ——【核心量表：PSWQ】**：
  * **PSWQ (Penn State Worry Questionnaire，宾州担忧问卷)**：
    * 字段：`PSWQ`（N = 198 有效，分值范围 16–80）。
    * **临床/认知意义**：在当代临床心理学与经颅磁刺激（TMS）神经调控领域，**担忧（Worry）与反刍（Rumination）同属“重复消极思维”（Repetitive Negative Thinking, RNT）的核心亚型**。两者在认知控制缺陷、默认网络（DMN）过度连接以及背外侧前额叶（DLPFC）低反应性上共享高度一致的神经生物学基础。PSWQ 是评估特质性重复负性思维的标杆量表。
  * **情绪调节问卷 (ERQ)**：`ERQ_REAP`（认知重评）与 `ERQ_SUP`（表达抑制），反刍思维常与低认知重评及高表达抑制共变。
* **Iowa 3T (UIowa JJ_TMS) ——【临床反刍、担忧与负性认知图式】**：
  * 虽未施测独立的 RRS 量表，但完整具备多个高信度的反刍与侵入性消极思维指标：
    * **DSM-5 横切量表 Item 16 (`dsm5crossad_16`)**（N = 94）：*“During the past two weeks, how often have you been bothered by unpleasant thoughts, urges, or images that repeatedly enter your mind?”*（评估反复侵入脑海的消极观念/强迫性反刍）。
    * **BDI-II 核心反刍/负性沉思条目**：`bdi_2`（Pessimism 悲观）、`bdi_5`（Guilty Feelings 内疚自责）、`bdi_7`（Self-Dislike 自厌）、`bdi_8`（Self-Criticalness 自我挑剔）。
    * **MADRS-SR Item 8 (`madrs_8`)**（N = 94）：*Pessimism*（悲观、内疚、对未来反复担忧）。
    * **GAD-7 担忧失控条目**：`gad_2`（无法停止或控制担忧，N = 95）与 `gad_3`（对各种事情过分担忧，N = 95）。
    * **STAI 特质担忧条目**：`stai_t9`（*“I worry too much over something that really doesn't matter”*，N = 94）。

### 3. Memory（记忆功能：工作记忆、延迟回忆与主观记忆抱怨）
* **Stanford (Causal Connectome) ——【客观计算机化记忆测验组合 (Test Battery)】**：
  * `Working_Memory_Norm` / `Working_Memory_Sten`（N = 193）：WebNeuro 工作记忆常模/十分分，基于空间工作记忆与多重记忆测验。
  * `Recall_Memory_Norm` / `Recall_Memory_Sten`（N = 196）：WebNeuro 延迟回忆与情景记忆常模/十分分。
  * **数字广度测验 (Digit Span)**：`digitsp`（顺背/倒背广度）、`digitot`（总分）、`digitsp_norm`（N = 193），评估工作记忆容量与瞬时言语记忆。
  * **连续追踪言语记忆测验 (Continuous Tracking Memory / Verbal Learning)**：
    * `ctmlearn_norm`（言语学习率，N = 195）
    * `ctmrec1_norm` ~ `ctmrec3_norm`（即刻言语回忆 Trial 1–3）
    * `ctmrec4_norm`（延迟言语回忆 Delayed Recall，N = 195）
    * `ctmerr4_norm`（延迟回忆错误率）
  * **WASI-II 矩阵推理**：`WASI_Matrix_T_Score`（N = 304），流体智力与非言语工作记忆。
  * **器质性/创伤后失忆病史**：`Amnesia`（是否曾经历记忆丧失/逆行顺行失忆）。
* **Iowa 3T (UIowa JJ_TMS) ——【主观记忆损害与功能障碍】**：
  * **DSM-5 横切量表 Item 15 (`dsm5crossad_15`)**（N = 93）：
    * *“During the past two weeks, how often have you been bothered by problems with memory (e.g., learning new information) or with location (e.g., finding your way home)?”*（直接评估学习新信息或定向相关的记忆障碍频率）。
  * **WHODAS 2.0 Item 3 (`whodas_3`)**（N = 94）：*“how much difficulty did you have in learning a new task, for example, learning how to get to a new place?”*（空间定向与程序/情节学习功能）。
  * **PCL-5 创伤记忆条目**（N = 48）：`pcl_1`（侵入性不自主创伤记忆）与 `pcl_8`（创伤关键细节遗忘/解离性记忆丧失）。
  * **注意**：Iowa 3T 中 NIH Toolbox 仅施测了 **Emotion Domain**，**未包含** Cognition Domain 的客观记忆测验（如 Picture Sequence Memory）。

### 4. Attention（注意功能：持续性注意、控制性注意、注意涣散与冲突抑制）
* **Stanford (Causal Connectome) ——【客观注意测验 + 实验范式】**：
  * `Sustained_Attention_Norm` / `Sustained_Attention_Sten`（N = 196）：持续性注意力常模/十分分，衡量警觉性维持。
  * `Controlled_Attention_Norm` / `Controlled_Attention_Sten`（N = 194）：控制性/选择性注意力常模/十分分。
  * **Go/No-Go 持续表现测验**：
    * `g2acck_norm`（抑制控制正确率，N = 194）
    * `g2avrtk_norm`（平均反应时，N = 194）
    * `g2sdrtk_norm`（反应时标准差，N = 193，**衡量个体内部注意力波动与稳定性 Intra-individual Variability**）
  * **MSIT (Multi-Source Interference Task) 实验范式**：整合 Stroop 与 Flanker 效应的注意冲突与认知控制经典任务（`MSIT_Done`, `Task_Score_MSIT`）。
* **Iowa 3T (UIowa JJ_TMS) ——【自评注意缺陷与结构化访谈诊断】**：
  * **全样本高信度自评注意缺陷条目**：
    * **PHQ-9 Item 7 (`phq_7`)**（N = 95）：*“Trouble concentrating on things, such as reading the newspaper or watching television”*（专注力减退）。
    * **BDI-II Item 19 (`bdi_19`)**（N = 94）：*Concentration Difficulty*（专注困难度：0 分正常至 3 分“无法集中注意力在任何事情上”）。
    * **BDI-II Item 13 (`bdi_13`)**（N = 94）：*Indecisiveness*（优柔寡断/注意决断困难）。
    * **MADRS-SR Item 5 (`madrs_5`)**（N = 94）：*Ability to Concentrate*（专注意识与收集思绪的能力）。
    * **WHODAS 2.0 Item 6 (`whodas_6`)**（N = 94）：*“how much difficulty did you have in concentrating on doing something for ten minutes?”*（专注维持 10 分钟的困难程度）。
    * **PCL-5 Item 19 (`pcl_19`)**（N = 48）：*“Having difficulty concentrating?”*。
    * **PANAS-SF Item 17 (`panas_17`)**（N = 94）：*Attentive*（专注/敏锐的积极注意状态）。
  * **M.I.N.I. 7.0.2 精神科临床诊断条目**：
    * `mini_a3f`（抑郁发作集中注意力与做决定困难，N = 27）。
    * `mini_n3d`（广泛性焦虑注意力涣散或脑子一片空白，N = 27）。

---

## 七、四维度跨数据集对照与建模策略

| 维度 | Stanford (Causal Connectome) | Iowa 3T (UIowa JJ_TMS) | 跨数据集建模与联合分析建议 |
| :--- | :--- | :--- | :--- |
| **Psychomotor** | • WebNeuro `Motor_Coordination_Norm` (194)<br>• WebNeuro `Processing_Speed_Norm` (197)<br>• 敲击 `tdomnk_norm` / 反应时 `chlrrtav_norm` | • PHQ-9 Item 8 (`phq_8`, 95)<br>• BDI-II Item 11 (`bdi_11`, 94)<br>• BDI-II Item 20 (`bdi_20`, 94)<br>• MADRS-SR Item 6 (`madrs_6`, 94) | • **临床条目对齐**：两数据集均可用 **PHQ-9 Item 8** 与 **BDI/BDI-II Item 11/20**。<br>• **客观测验**：Stanford 特有完整的 WebNeuro 动作反应时与敲击数据。 |
| **Rumination / RNT** | • **PSWQ (宾州担忧问卷总分, 198)**<br>• ERQ 认知重评与表达抑制 | • DSM-5 横切 Item 16 (`dsm5crossad_16`, 94)<br>• BDI-II Items 2, 5, 7, 8 (悲观/内疚/自责)<br>• MADRS-SR Item 8 (`madrs_8`, 94)<br>• GAD-7 Items 2, 3 (`gad_2`, `gad_3`, 95) | • 均具备出色的**重复消极思维（RNT）**度量。<br>• Stanford 推荐使用 **PSWQ**；Iowa 3T 推荐使用 **`dsm5crossad_16` + BDI-II Items 2/5/7/8 + GAD Items 2/3** 合成 RNT 指数。 |
| **Memory** | • WebNeuro `Working_Memory_Norm` (193)<br>• WebNeuro `Recall_Memory_Norm` (196)<br>• 数字广度 `digitsp`/`digitot` (193)<br>• 言语学习/延迟回忆 `ctmlearn`/`ctmrec4` (195) | • DSM-5 横切 Item 15 (`dsm5crossad_15`, 93)<br>• WHODAS Item 3 (`whodas_3`, 94)<br>• PCL-5 Items 1, 8 (创伤遗忘/侵入) | • **客观 vs 主观分层**：Stanford 为**客观实验神经心理测验**（工作记忆与言语延迟回忆）；Iowa 3T 为**主观日常记忆障碍与学习损害自评**。两者联合可作为“客观认知 vs 主观认知抱怨”的双视角模型。 |
| **Attention** | • WebNeuro `Sustained_Attention_Norm` (196)<br>• WebNeuro `Controlled_Attention_Norm` (194)<br>• Go/No-Go 稳定性 `g2sdrtk_norm` (193)<br>• MSIT 冲突任务 | • PHQ-9 Item 7 (`phq_7`, 95)<br>• BDI-II Item 19 (`bdi_19`, 94)<br>• MADRS-SR Item 5 (`madrs_5`, 94)<br>• WHODAS Item 6 (`whodas_6`, 94)<br>• PANAS Item 17 (`panas_17`, 94) | • **完全同质条目**：直接提取 **PHQ-9 Item 7** 与 **BDI/BDI-II Item 19**，两数据集跨样本注意力集中困难分布完全对齐。<br>• Stanford 可补充持续注意与反应时稳定性。 |


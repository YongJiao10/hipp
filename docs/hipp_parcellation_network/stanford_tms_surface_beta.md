# Stanford 三网络左右海马 Native-space surface beta

这是 Stanford 当前的探索性 beta 分析入口。旧的 Standard/MNI-space beta 结果
已归档；Native 分支尚未晋升为编号正式流程或冻结结果。

## 固定分析定义

- 网络：`Control`、`Default`、`VentralAttention`，显示为 FPCN、DMN、VAN。
- 健康组：`NTHC + TEHC`；病人组：`NTS + TIS`。
- 当前探索 cohort：966 个 `subject × site × condition` top-3 行。
- 海马：左、右分别统计；不生成 `bilateral`。
- HippUnfold parcellation：individual template matching 和固定 HCP151，各自保留。
- 测量：只执行 `volume → hippocampal surface`；不执行 `surface → volume`。
- con：Native-space Float32 `con_0001.nii.gz`，不 GrayZ、不重采样。
- Native 表面：Stanford HippUnfold den-512 `space-corobl` inner/midthickness/outer。
- beta：在 L/R 海马的 cluster 1/2/3 上按 midthickness vertex area 加权。
- 同一 subject/network/target hemisphere 的多个 site 先等权平均，同时保留 site-level 表。

TMS network 和 group 已在 `native_jobs.csv` 中冻结。`targetBrainEnvLocMNI` 只用于
之前的 Stanford site/network 选择；Native beta 提取本身不再把 MNI con 与个体表面
拼接。

## 输入与转换结果

Native con：

```text
outputs_stanford/inputs/stanford/tmsfmri_fmriprep/
group_con_results_NativeSpace/group_con_data_6P/
```

已完成的 Native volume-to-surface 结果：

```text
outputs_native_surface_comparison/
  stanford_native_standard_bad_compare_20260818/stanford/
```

该目录的 `native_jobs.csv` 有 966 行，metrics 和 bad-vertex masks 均已生成，
`run_manifest.json` 记录 mapping errors=0。Native con 的背景包含精确 0，因此
转换阶段同时生成 `finite` 和 `nonzero` 两种 ROI；当前探索提取使用 `nonzero`，
`finite` 只作为敏感性记录。

## 探索性 Native beta 提取

```bash
/opt/miniconda3/envs/py314/bin/python \
  scripts/exploratory_beta/run_native_surface_beta.py \
  --dataset stanford \
  --run-id stanford_native_surface_beta_20260818_v2 \
  --roi-policy nonzero
```

默认输出：

```text
outputs_stanford/work/tms_hipp_surface_beta_native/runs/
  stanford_native_surface_beta_20260818_v2/
```

关键表：

- `cohort_flow.csv`：966 行 Native job 及提取状态；
- `parcellation_exclusions.csv`：template matching 固定 roster 之外的 32 个
  site-parcellation 排除记录；
- `site_level_beta.csv`：site × parcellation × hippocampal hemisphere × cluster；
- `subject_network_beta.csv`：subject/network/target-hemi 等权 site 平均；
- `extraction_summary.json`、`run_manifest.json`：空间、ROI、计数和方法契约。

## 模型

```bash
Rscript scripts/exploratory_beta/stanford_tms_surface_model.R \
  outputs_stanford/work/tms_hipp_surface_beta_native/runs/stanford_native_surface_beta_20260818_v2
```

分别拟合 healthy/patient × template matching/HCP151 四个预设探索分支，并同时保存
subject-network primary 和 site-level sensitivity：

```text
beta_mean ~ site_network * cluster * hippocampal_hemi
          + target_hemi * cluster
          + target_hemi * hippocampal_hemi
random = ~1 | subject / hippocampal_hemi
```

网络 pairwise contrasts 在每个 group × parcellation × source branch 内 Holm 校正；
不拟合健康/病人交互，也不加入 `site_network × target_hemi`。

## PNG 刷新

```bash
/opt/miniconda3/envs/py314/bin/python \
  scripts/exploratory_beta/stanford_tms_five_figures.py \
  --run-dir outputs_stanford/work/tms_hipp_surface_beta_native/runs/stanford_native_surface_beta_20260818_v2
```

PNG 和支持表写入该 run 的 `five_figures/`；默认不生成 PDF，显式加入 `--pdf` 才额外
导出 PDF。热图左列固定为左海马，右列固定为右海马；所有文字使用新的 Native run 数据，
旧 Standard-space 图位于归档目录。

Healthy profile figure additionally performs uncorrected Welch independent-samples tests with
VAN as the reference, comparing VAN against Default/DMN and Control/FPCN for every
parcellation × hippocampal hemisphere × cluster. Significant VAN-higher comparisons use red
brackets and `*`/`**`/`***`; significant VAN-lower comparisons use blue brackets. Non-significant
comparisons have no bracket or label. Results are written to
`healthy_van_vs_fpcn_dmn_ttests.csv`.

Template matching labels follow the fixed 194-person QC-passed roster. The
five HippUnfold-only subjects `1055`, `2001`, `3046`, `3107`, and `4117` are
recorded in `parcellation_exclusions.csv` and are excluded only from the
individual template-matching branch; HCP151 remains available for them.

## 旧结果归档

```text
outputs_stanford/archive/20260818_standard_space_beta_replaced_by_native/
```

其中保存旧 run、旧 Argon/MNI 入口代码和归档说明。归档内容仅用于 provenance，
不得与新的 Native beta 结果混合；当前 Native 结果也不得在未完成正式晋升前
表述为冻结或可直接报告的正式结果。

## Stanford TMS 个体化 Kong2022 7-Network 映射与 LSS 共享

Stanford TMS 靶点（`targetBrainEnvLocWorld`，LAS 坐标系经 X 轴反转对齐至 RAS+ 表面空间）
映射至每个受试者 Kong2022 个体化 7 网络分区（`canonical7` 体系：Default, Visual, Somatomotor, DorsalAttention, VentralAttention, Control, Language）：

- 提取脚本：[`scripts/exploratory_beta/map_stanford_tms_sites_to_kong2022.py`](file:///home/yojiao/hpchome/hipp/scripts/exploratory_beta/map_stanford_tms_sites_to_kong2022.py)
- 本地副本：[`outputs_stanford/present/tms_site_networks/`](file:///home/yojiao/hpchome/hipp/outputs_stanford/present/tms_site_networks/)
- LSS 汇总产物：
  - CSV: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/stanford_tms_site_kong2022_individual_network_summary.csv` (1,663 行总计，1,573 行成功映射覆盖 151 名被试及 11 个 TMS 靶点；9 名缺失解剖 T1w 被试显式 fail-closed 记录原因并标记为 unmapped)
  - JSON: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/stanford_tms_site_kong2022_individual_network_summary.json`

## Stanford HippUnfold 表面文件共享至 LSS

Stanford 全部 199 名受试者的 HippUnfold 表面文件（99 个文件/被试，共 19,701 个文件，2,195.25 MB）
已按照一层目录扁平结构同步至共享存储，便于直接读取与批处理：

- 传输脚本：[`scripts/core_workflow/copy_stanford_hippunfold_surfaces_to_lss.py`](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/copy_stanford_hippunfold_surfaces_to_lss.py)
- LSS 目标路径：`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/HippoUnfold/`
- 结构规范：`HippoUnfold/sub-<id>/<99_files>`（单层扁平，无嵌套子目录）
- 传输收据：`/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/HippoUnfold/copy_receipt.json`


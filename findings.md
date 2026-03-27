# Findings

## Confirmed
- 本地目录当前只有论文 PDF 与一张流程说明图，没有现成代码。
- 当前机器可解析 `MPM619.local -> 192.168.0.113`。
- 当前机器已成功连接远端 `ssh yojiao@192.168.0.113`。
- 论文/教程兼容流程里，rsfMRI 示例默认海马 surface density 为 `2mm`。
- 用户已明确默认 density 采用 `2mm`。
- 远端外接盘路径为 `/Volumes/Elements`，HCP 数据根目录为 `/Volumes/Elements/HCP-YA-2025`。
- 已发现候选被试 `100610`，同时具备结构包和 7T rsfMRI volume。
- 已从远端 zip 中只提取 `100610/T1w/T1w_acpc_dc_restore.nii.gz` 与 `100610/T1w/T2w_acpc_dc_restore.nii.gz`，并复制 7T concat 与 brain mask。
- 已下载官方 `Schaefer2018_400Parcels_7Networks_order.dlabel.nii` 与 `...FSLMNI152_2mm.nii.gz`。
- 本机 `wb_command` 可用路径为 `/Applications/wb_view.app/Contents/usr/bin/wb_command`。
- 本机无 `docker`，无已安装的 `hippunfold`。
- conda 频道当前可见 `hippunfold 2.0.0`，但实际安装后 `hippunfold --version` 输出为 `1.5.2-pre.2`。
- 当前被试本地输入经脚本自动检测为 `functional_mode = volume`，并被规则性阻断。
- `HippoMaps 0.1.17` 已装入 `hippo` 环境，但顶层 `import hippomaps` 会触发 Qt/VTK crash。
- 可绕过顶层导入，直接按文件路径加载 `/opt/miniconda3/envs/hippo/lib/python3.11/site-packages/hippomaps/utils.py`。
- 参考 `/Users/jy/Downloads/workbench_plotting_macbook` 后已确认：本机 `wb_command` 需通过 `arch -x86_64` 调用，`-help` 现可正常运行。
- 已完成单一分析输入 staging：`data/hippunfold_input/sub-100610/{anat,func}`。
- 已完成 volume-based Schaefer 参考提取，产出位于 `outputs/100610/reference/`，包含 400 parcel 与 7 network 时序。
- 结构像与 BOLD 头信息对齐良好：T1/T2 为 0.7mm，BOLD 为 1.6mm，且 qform/sform 均为 4。
- 已补充海马 surface 时序采样脚本与左右海马 label 合并回采样脚本，准备接在 HippUnfold 输出之后运行。
- HippUnfold 现已能走到 nnUNet 模型下载步骤，模型正下载到 `/tmp/hippunfold_cache/model/`。
- 用户已明确：当前仅在本地 macOS 做流程打通，正式运行目标平台是 HPC SGE。
- 用户已明确：HPC 正式版 `nnUNet/HippUnfold` 走 GPU，不做 CPU 兼容与 CPU 正式运行路线。
- 已确认本地 macOS 上 `nnUNet_predict` 的默认 multiprocessing `spawn` 会触发 `PicklingError`，不适合作为本地测试默认路径。
- 已实现按系统自动调整的兼容逻辑：Darwin/macOS 下自动切到 `NNUNET_DISABLE_MULTIPROCESSING=1`、单进程低内存路径；Linux/HPC 保留正常并行语义。
- 本地测试已进一步降到 `fold-0` 单折，以尽快获取功能 label；正式 HPC 版仍应恢复完整折数并使用 GPU。

## Constraints
- 远端数据在远端 Mac 的外接盘，不在本地。
- 只能最小拷贝单被试所需数据。
- 任何环境不确定性都必须显式汇报。
- 未经用户批准，不得从 `CIFTI-first` 路线切换到 `volume-based neocortical reference`。
- 当前实际可运行的 HippUnfold 密度口径仍为 `0p5mm / 1mm / 2mm / unfoldiso`，不得再按 `512` 继续实现。
- `102311` 的 full-fold nnUNet 推理本身完成，失败点在 HippUnfold `postproc_boundary_vertices`，不是 nnUNet 直接崩溃。
- `102311` 右侧海马 4 个 `*_sdt.nii.gz` 体数据正常，但对应 4 个 `*.shape.gii` 被 `wb_command -volume-to-surface-mapping` 映射成全零。
- `sub-102311_hemi-R_space-corobl_label-hipp_midthickness.surf.gii` 与 corobl 分割盒在 `y/z` 方向基本一致，但 `x` 方向整体偏移约 `30.6 mm`，等于 `102 * 0.3 mm`。
- 对当前曲面点做 `x -= shape_x * abs(affine_x)` 的负-x 校正后，`6006/6006` 顶点重新落回体内，直接三线性采样可恢复非零距离场。
- `render_native_surface_label_map.py` 的 folded/native 视图可直接复用 HippUnfold 的 `space-T2w` midthickness surface 与 `.label.gii/.npy` 标签，不依赖更脆弱的 volume 回投。
- `surface_labels_to_volume.py` 在 `py314` 下失败的第一层原因是它直接加载 `hippo` 环境的 `hippomaps/utils.py`；已在 post pipeline 中改为显式使用 `/opt/miniconda3/envs/hippo/bin/python` 执行该步骤。
- `surface_labels_to_volume.py` 当前仍有更深一层路径/空间假设：`hippomaps.utils.surface_to_volume` 会强行查找 `sub-*/coords/sub-*_space-T2w_*_laplace_coords.nii.gz`，但当前 HippUnfold 产物实际坐标场位于 `work/sub-*/coords/` 且命名为 `space-corobl`，所以会继续 `IndexError`。
- 本地 macOS 的兼容补丁只能视为测试机适配，不得把它误当成 HPC 正式运行默认参数。
- HPC 正式版不需要为 CPU fallback 保留额外兼容逻辑。

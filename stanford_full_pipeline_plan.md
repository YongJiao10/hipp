# Stanford 全流程执行计划：路径模板优先，symlink 仅作窄桥接

> **给 Claude / 后续执行 agent 的硬性说明**
>
> - 不要按旧方案创建一棵完整的 `data/stanford_input/` symlink tree 来伪装 HCP 命名。
> - 先小幅修改 `scripts/core_workflow/`，让核心 workflow 通过 CLI 路径模板读取 Stanford 原始命名。
> - 所有 Python repo 任务都用 `/home/yojiao/.conda/envs/py314/bin/python`，不要用裸 `python`。
> - 任何代码行为、路径、缓存、launcher、环境或 workflow 输出变更，都必须同步更新文档。
> - 不要删除任何文件。若确实需要删除，必须先列出精确目标路径并等待用户确认。
> - 不要访问或浏览 archive 内容，除非用户在当前会话明确授权。
> - 并发上限是 10。本计划所有批处理默认不超过 4；Workbench 密集步骤不超过 2。

## 目标

运行 Stanford 数据的海马功能分区全流程：

`HippUnfold -> hippocampal surface sampling/QC/smoothing -> cortex reference -> FC -> gradients -> group atlas -> visualization/matrix`

数据与现有资产：

- Stanford fMRIPrep 数据 root：
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P`
- 预计 199 subjects，BIDS-style subject 目录形如 `sub-2001`。
- Stanford 皮层个体化标签已在 `ArealMSHBM_Kong2022/`，但必须验证是否能直接满足 RouteA flat label contract。
- 输出统一放到 repo 内 `outputs_stanford/`。

## 核心判断

原 plan 的 symlink tree 方案能短期绕过 HCP 命名硬编码，但不是最佳长期方案。更好的方案是让 core workflow 接受路径模板：

- `--dtseries-template`：解决 `03_01`、`03_02`、`04_00`、`04_01` 的 CIFTI dtseries 文件名差异。
- `--bold-template`：解决 `02_01` 的 volume BOLD 文件名差异。
- 继续使用已有 `--left-cortex-surface-template` / `--right-cortex-surface-template`：解决 cortex surface 文件名差异。
- RouteA label 如果实际目录不符合 `sub-XXXX/lh_labels.txt` 和 `sub-XXXX/rh_labels.txt`，只对这一个窄接口做 symlink 或转换桥接。

不要为 dtseries、surface、volume BOLD 建全量 symlink tree。这样可以保持 provenance 清楚，避免把 Stanford 数据伪装成 HCP，并避免无谓解压 199 个 `.nii.gz` volume。

## Stanford 与当前 HCP 命名差异

| 用途 | 当前 core workflow 默认 | Stanford 实际路径模板 | 处理方式 |
|---|---|---|---|
| Cortex dtseries | `{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii` | `{data_root}/{sub_subject}/func/{sub_subject}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` | 新增 `--dtseries-template` |
| Volume BOLD | `{bold_root}/{subject}/func_uncompressed.nii` | `{bold_root}/{sub_subject}/func/{sub_subject}_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz` | 新增 `--bold-template`，先尝试直接读 `.nii.gz` |
| Cortex midthickness surface | `{data_root}/{sub_subject}/anat/{sub_subject}_hemi-{H}_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii` | `{data_root}/{sub_subject}/anat/{sub_subject}_hemi-{H}_space-fsLR_den-32k_midthickness.surf.gii` | 已有 surface template CLI，可直接覆盖 |
| RouteA individualized labels | `{route_a_root}/{sub_subject}/lh_labels.txt` and `rh_labels.txt` | 待验证 `ArealMSHBM_Kong2022/` 实际布局 | 必要时只桥接 label 目录 |

模板变量约定：

- `{data_root}`：`--data-root` 的绝对路径。
- `{bold_root}`：`--bold-root` 的绝对路径。
- `{subject}`：不带 `sub-` 的 subject id，例如 `2001`。
- `{sub_subject}`：带 `sub-` 的 subject id，例如 `sub-2001`。
- `{hemi}`：`L` 或 `R`。
- `{hemi_lower}`：`l` 或 `r`。

## Phase A: 先修改 core workflow 支持路径模板

这部分是 Claude 需要先执行的代码/doc/test 工作。完成后再跑 Stanford 数据。

### A0. 环境与仓库预检

在任何 Python repo 任务前，确认将使用 py314：

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
```

期望输出：

```text
/home/yojiao/.conda/envs/py314/bin/python
```

查看当前 git 状态，只读检查即可：

```bash
git status --short
```

如果工作树已有用户未提交修改，不要回滚。只编辑本计划列出的文件；如果同一文件已有用户改动，先读懂再合并。

### A1. 新增统一路径模板 helper

新增文件：

- `scripts/core_workflow/lib/path_templates.py`

建议实现：

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def normalize_subject(subject: str) -> str:
    """Return subject id without the sub- prefix."""
    return str(subject).replace("sub-", "")


def format_subject_path_template(
    template: str,
    *,
    subject: str,
    hemi: str | None = None,
    **roots: str | Path,
) -> Path:
    """Format a per-subject path template used by core workflow entrypoints.

    Supported variables:
    - subject: id without sub-, e.g. 2001
    - sub_subject: id with sub-, e.g. sub-2001
    - hemi: L/R when provided, otherwise empty string
    - hemi_lower: l/r when provided, otherwise empty string
    - any named root passed in roots, e.g. data_root or bold_root
    """
    subject_id = normalize_subject(subject)
    hemi_value = "" if hemi is None else str(hemi)
    values: dict[str, str] = {
        name: str(Path(value).resolve())
        for name, value in roots.items()
    }
    values.update(
        {
            "subject": subject_id,
            "sub_subject": f"sub-{subject_id}",
            "hemi": hemi_value,
            "hemi_lower": hemi_value.lower(),
        }
    )
    try:
        rendered = str(template).format(**values)
    except KeyError as exc:
        available = ", ".join(sorted(values))
        raise KeyError(
            f"Unknown path template variable {exc!s}; available variables: {available}"
        ) from exc
    return Path(rendered).resolve()
```

Why this helper:

- Avoid copy-pasting template formatting logic into four entrypoints.
- Preserve existing default behavior by setting defaults to the current HCP paths.
- Make Stanford paths explicit in CLI commands and manifests.

### A2. Add tests for template formatting

Create:

- `tests/core_workflow/test_path_templates.py`

Suggested tests:

```python
from pathlib import Path

import pytest

from scripts.core_workflow.lib.path_templates import (
    format_subject_path_template,
    normalize_subject,
)


def test_normalize_subject_removes_sub_prefix():
    assert normalize_subject("sub-2001") == "2001"
    assert normalize_subject("2001") == "2001"


def test_format_subject_path_template_supports_stanford_dtseries():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii",
        data_root="/tmp/data",
        subject="sub-2001",
    )
    assert path == Path(
        "/tmp/data/sub-2001/func/sub-2001_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    )


def test_format_subject_path_template_supports_legacy_bold_default():
    path = format_subject_path_template(
        "{bold_root}/{subject}/func_uncompressed.nii",
        bold_root="/tmp/bold",
        subject="sub-2001",
    )
    assert path == Path("/tmp/bold/2001/func_uncompressed.nii")


def test_format_subject_path_template_supports_hemi_surface():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii",
        data_root="/tmp/data",
        subject="2001",
        hemi="L",
    )
    assert path == Path(
        "/tmp/data/sub-2001/anat/sub-2001_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
    )


def test_format_subject_path_template_reports_unknown_variable():
    with pytest.raises(KeyError, match="Unknown path template variable"):
        format_subject_path_template(
            "{data_root}/{missing}/{sub_subject}.nii",
            data_root="/tmp/data",
            subject="2001",
        )
```

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_path_templates.py -q
```

Expected: the new tests pass.

### A3. Modify `02_01_prepare_group_inputs.py`

File:

- `scripts/core_workflow/02_01_prepare_group_inputs.py`

Required behavior:

- Add `DEFAULT_BOLD_TEMPLATE = "{bold_root}/{subject}/func_uncompressed.nii"` near existing defaults.
- Import `format_subject_path_template`.
- Add CLI argument:

```python
parser.add_argument(
    "--bold-template",
    default=DEFAULT_BOLD_TEMPLATE,
    help=(
        "Per-subject BOLD volume template. Variables: {bold_root}, {subject}, "
        "{sub_subject}. Default preserves the legacy HCP layout."
    ),
)
```

- Change `run_raw_sampling(...)` signature to accept `bold_template: str`.
- Replace the current hard-coded line:

```python
bold = Path(bold_root) / subject / "func_uncompressed.nii"
```

with:

```python
bold = format_subject_path_template(
    bold_template,
    bold_root=bold_root,
    subject=subject,
)
```

- When `run_stage(...)` dispatches `run_raw_sampling`, pass `args.bold_template`.
- Do not change `load_subjects(...)` unless needed for tests. For Stanford, always pass `--subjects` during smoke and `--subjects-manifest` during full runs so subject discovery does not depend on the legacy `{bold_root}/{subject}` directory pattern.

Add or update tests if the repo already has CLI/parser tests. At minimum, compile:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/02_01_prepare_group_inputs.py
```

### A4. Modify `03_01_build_kong2022_reference.py`

File:

- `scripts/core_workflow/03_01_build_kong2022_reference.py`

Required behavior:

- Add:

```python
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
```

- Import `format_subject_path_template`.
- Add CLI argument:

```python
parser.add_argument(
    "--dtseries-template",
    default=DEFAULT_DTSERIES_TEMPLATE,
    help=(
        "Per-subject dtseries template. Variables: {data_root}, {subject}, "
        "{sub_subject}. Default preserves the legacy HCP layout."
    ),
)
```

- Add `dtseries_template: str` to `run_subject(...)`.
- Replace the hard-coded command argument:

```python
str(data_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii")
```

with:

```python
str(
    format_subject_path_template(
        dtseries_template,
        data_root=data_root,
        subject=subject,
    )
)
```

- Pass `args.dtseries_template` from `main()` into each `run_subject(...)` call.

Compile:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/03_01_build_kong2022_reference.py
```

### A5. Modify `03_02_build_kong2022_routea_reference.py`

File:

- `scripts/core_workflow/03_02_build_kong2022_routea_reference.py`

Make the same `--dtseries-template` change as A4. Keep the default legacy path identical to current behavior.

Compile:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/03_02_build_kong2022_routea_reference.py
```

### A6. Modify `04_00_materialize_cortex_regressed_surface_timeseries.py`

File:

- `scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py`

Required behavior:

- This step is used for RouteA cortex-regressed variants. Stanford routeb-only execution may not use it immediately, but it currently has the same HCP dtseries hard-code and should be kept consistent.
- Add `DEFAULT_DTSERIES_TEMPLATE` with the legacy default.
- Add `--dtseries-template` to the parser.
- Change `dtseries_path(...)` to either accept `dtseries_template` or remove it and call `format_subject_path_template(...)` in `process_subject(...)`.
- Keep existing `--left-cortex-surface-template` and `--right-cortex-surface-template` behavior.
- Prefer replacing the local `format_subject_template(...)` with the shared helper only if the edit remains small. If replacing risks broad churn, leave surface template formatting as-is and use the new helper only for dtseries.

Compile:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py
```

### A7. Modify `04_01_materialize_kong2022_roi_fc.py`

File:

- `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py`

Required behavior:

- Add `DEFAULT_DTSERIES_TEMPLATE` with the legacy default.
- Add parser argument:

```python
parser.add_argument("--dtseries-template", default=DEFAULT_DTSERIES_TEMPLATE)
```

- Replace `dtseries_path(data_root, subject)` or change it to:

```python
def dtseries_path(data_root: Path, subject: str, dtseries_template: str) -> Path:
    return format_subject_path_template(
        dtseries_template,
        data_root=data_root,
        subject=subject,
    )
```

- Pass `args.dtseries_template` through to the function that loads cortex dtseries.
- Keep existing surface template overrides. Stanford commands will use:

```text
{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_midthickness.surf.gii
{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_midthickness.surf.gii
```

Compile:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py
```

### A8. Update core workflow documentation

File:

- `scripts/core_workflow/README.md`

Update the "Entrypoint arguments" section so it states:

- `02_01_prepare_group_inputs.py` accepts `--bold-template`.
- `03_01_build_kong2022_reference.py`, `03_02_build_kong2022_routea_reference.py`, `04_00_materialize_cortex_regressed_surface_timeseries.py`, and `04_01_materialize_kong2022_roi_fc.py` accept `--dtseries-template`.
- Templates support `{data_root}`, `{bold_root}`, `{subject}`, `{sub_subject}`, `{hemi}`, `{hemi_lower}` where relevant.
- Defaults preserve existing HCP behavior.
- Stanford and future non-HCP datasets should prefer templates over broad symlink trees.

This README update is required by the repository Doc Sync Rule.

### A9. Implementation verification before data runs

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_path_templates.py -q
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/lib/path_templates.py \
  scripts/core_workflow/02_01_prepare_group_inputs.py \
  scripts/core_workflow/03_01_build_kong2022_reference.py \
  scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py \
  scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py
```

Optional broader regression:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow -q
```

If broad tests fail for unrelated pre-existing reasons, record the failing test names and continue only if the new path-template tests and py_compile pass.

## Phase B: Stanford data preflight

Use these shell variables for all Stanford commands:

```bash
PY314=/home/yojiao/.conda/envs/py314/bin/python
STANFORD_ROOT=/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P
STANFORD_OUT=outputs_stanford
HIPPO_BATCH=${STANFORD_OUT}/inputs/stanford/hippunfold/dense_corobl_batch
KONG_ROOT=${STANFORD_OUT}/inputs/stanford/kong2022
MANIFEST=${KONG_ROOT}/manifests/stanford_hippocampus_fmri_qc_pass_t1only_latest.txt
SURFACE_ROOT=${KONG_ROOT}/shared_inputs/surface_151
ROUTEA_ROOT=${KONG_ROOT}/shared_inputs/routea_151
ROUTEB_ROOT=${KONG_ROOT}/shared_inputs/routeb_151
TEMPLATE_ROOT=${KONG_ROOT}/templates/hipp_group_average_t1only_151
ROUTEA_LABEL_ROOT=${KONG_ROOT}/cortex_references/routea_areal_mshbm
DT_TEMPLATE='{data_root}/{sub_subject}/func/{sub_subject}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
BOLD_TEMPLATE='{bold_root}/{sub_subject}/func/{sub_subject}_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz'
LEFT_SURF_TEMPLATE='{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_midthickness.surf.gii'
RIGHT_SURF_TEMPLATE='{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_midthickness.surf.gii'
```

Create output directories:

```bash
mkdir -p \
  "${HIPPO_BATCH}" \
  "${KONG_ROOT}/manifests" \
  "${SURFACE_ROOT}" \
  "${ROUTEA_ROOT}" \
  "${ROUTEB_ROOT}" \
  "${TEMPLATE_ROOT}" \
  "${STANFORD_OUT}/work"
```

Validate HippUnfold environment before direct HippUnfold runs:

```bash
source /opt/miniconda3/bin/activate hippo2
which hippunfold
hippunfold --version
hippunfold --modality T2w --output_density 512 --help | grep 512
```

Required:

- `which hippunfold` returns `/opt/miniconda3/envs/hippo2/bin/hippunfold`
- `hippunfold --version` returns `2.0.0`
- help command succeeds and shows `512`

If the gate fails, repair exactly as documented in `AGENTS.md`:

```bash
CONDA_SAFETY_CHECKS=disabled conda install -y -n hippo2 -c khanlab -c conda-forge -c bioconda khanlab::hippunfold=2.0.0=py_0 --force-reinstall
```

## Phase C: Validate required Stanford source files

Use bounded directory operations only. Do not run unbounded `find` or recursive `ls`.

For smoke subject `2001`, verify the expected inputs:

```bash
test -f "${STANFORD_ROOT}/sub-2001/func/sub-2001_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
test -f "${STANFORD_ROOT}/sub-2001/func/sub-2001_task-rest_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
test -f "${STANFORD_ROOT}/sub-2001/anat/sub-2001_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
test -f "${STANFORD_ROOT}/sub-2001/anat/sub-2001_hemi-R_space-fsLR_den-32k_midthickness.surf.gii"
test -d "${STANFORD_ROOT}/sub-2001/anat"
```

If any command fails, stop and inspect with bounded listing:

```bash
timeout 60 find "${STANFORD_ROOT}/sub-2001/func" -maxdepth 1 2>/dev/null | head -40
timeout 60 find "${STANFORD_ROOT}/sub-2001/anat" -maxdepth 1 2>/dev/null | head -40
```

Do not proceed by guessing alternate filenames. Update templates only after confirming actual filenames.

## Phase D: RouteA label contract

`03_02` expects:

```text
${ROUTEA_LABEL_ROOT}/sub-2001/lh_labels.txt
${ROUTEA_LABEL_ROOT}/sub-2001/rh_labels.txt
```

Before creating any bridge, inspect `ArealMSHBM_Kong2022/` with bounded commands:

```bash
timeout 60 find ArealMSHBM_Kong2022 -maxdepth 2 2>/dev/null | head -80
```

If it already has `sub-XXXX/lh_labels.txt` and `sub-XXXX/rh_labels.txt`, set:

```bash
ROUTEA_LABEL_ROOT=ArealMSHBM_Kong2022
```

If not, create a narrow RouteA label bridge under:

```text
outputs_stanford/inputs/stanford/kong2022/cortex_references/routea_areal_mshbm/
```

Bridge rules:

- Only bridge `lh_labels.txt` and `rh_labels.txt`.
- Do not bridge dtseries, cortex surfaces, or BOLD volumes.
- Prefer symlinks if source files already have the expected text format.
- If source format differs, write a small conversion script and document it in `scripts/core_workflow/README.md` or a Stanford run note.
- Do not delete or overwrite existing bridge files without user confirmation.

Validate one subject:

```bash
test -f "${ROUTEA_LABEL_ROOT}/sub-2001/lh_labels.txt"
test -f "${ROUTEA_LABEL_ROOT}/sub-2001/rh_labels.txt"
head -5 "${ROUTEA_LABEL_ROOT}/sub-2001/lh_labels.txt"
head -5 "${ROUTEA_LABEL_ROOT}/sub-2001/rh_labels.txt"
```

Expected: one integer-like label vector per file, compatible with `scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py`.

## Phase E: Smoke test with `sub-2001`

Run exactly one subject first. Do not launch full 199-subject jobs until the smoke path completes through at least `04_02`.

### E1. HippUnfold smoke

```bash
"${PY314}" scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  2001 \
  "${STANFORD_ROOT}" \
  "${HIPPO_BATCH}/sub-2001/hippunfold"
```

Expected output root:

```text
outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch/sub-2001/hippunfold/sub-2001/
```

Check strict density evidence from output paths, not only command args:

```bash
test -f "${HIPPO_BATCH}/sub-2001/hippunfold/sub-2001/surf/sub-2001_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
test -f "${HIPPO_BATCH}/sub-2001/hippunfold/sub-2001/surf/sub-2001_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii"
test -f "${HIPPO_BATCH}/sub-2001/hippunfold/sub-2001/qc/sub-2001_hemi-L_desc-unetf3d_dice.tsv"
test -f "${HIPPO_BATCH}/sub-2001/hippunfold/sub-2001/qc/sub-2001_hemi-R_desc-unetf3d_dice.tsv"
```

If density evidence disagrees with command parameters, treat consumed output assets as authoritative and stop to report the mismatch.

### E2. Prepare hippocampal group inputs smoke

First try direct `.nii.gz` BOLD input. `sample_hipp_surface_timeseries(...)` passes `bold` directly to Workbench `-volume-to-surface-mapping`; Workbench commonly accepts `.nii.gz`, but this must be empirically verified here.

```bash
"${PY314}" scripts/core_workflow/02_01_prepare_group_inputs.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --route-root "${KONG_ROOT}" \
  --shared-root "${SURFACE_ROOT}" \
  --template-root "${TEMPLATE_ROOT}" \
  --bold-root "${STANFORD_ROOT}" \
  --bold-template "${BOLD_TEMPLATE}" \
  --subjects 2001 \
  --jobs 1
```

Expected files:

```bash
test -f "${SURFACE_ROOT}/sub-2001/surface/raw/surface_sampling_summary.json"
test -f "${SURFACE_ROOT}/sub-2001/surface/tsnr_gate/sub-2001_hemi-L_tsnr_gate_summary.json"
test -f "${SURFACE_ROOT}/sub-2001/surface/2mm/sub-2001_hemi-L_timeseries.npy"
test -f "${SURFACE_ROOT}/sub-2001/surface/4mm/sub-2001_hemi-R_timeseries.npy"
```

If `.nii.gz` fails:

- Do not immediately unpack all 199 subjects.
- Create a smoke-only uncompressed BOLD copy under `outputs_stanford/tmp_bold/sub-2001/func_uncompressed.nii`.
- Rerun `02_01` with `--bold-root outputs_stanford/tmp_bold --bold-template "{bold_root}/{subject}/func_uncompressed.nii"` for smoke.
- If uncompressed BOLD is required, update this plan with a batch conversion step under `outputs_stanford/tmp_bold/`. Do not write decompressed files into source data.

### E3. Build smoke QC manifest

```bash
"${PY314}" scripts/core_workflow/02_02_build_qc_manifest.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --output-manifest "${MANIFEST}" \
  --subjects 2001
```

Check:

```bash
test -f "${MANIFEST}"
cat "${MANIFEST}"
```

Expected: manifest contains `2001` or `sub-2001` in the format accepted by downstream scripts. Existing scripts strip `sub-` when needed.

### E4. RouteB cortex reference smoke

Find the maintained Kong2022 dlabel path before running. If unknown, search only bounded paths and do not browse archives:

```bash
timeout 60 find outputs_hcp software data scripts docs -maxdepth 4 -name "*Kong*2022*dlabel*.nii" 2>/dev/null | head -40
```

Set:

```bash
KONG2022_DLABEL=/absolute/path/to/kong2022.dlabel.nii
```

Then run:

```bash
"${PY314}" scripts/core_workflow/03_01_build_kong2022_reference.py \
  --shared-root "${ROUTEB_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 1
```

Expected:

```bash
test -f "${ROUTEB_ROOT}/sub-2001/reference/kong2022/reference_summary.json"
test -f "${ROUTEB_ROOT}/sub-2001/reference/kong2022/stage_manifest.json"
```

### E5. RouteA cortex reference smoke

Only run after RouteA label contract is verified.

```bash
"${PY314}" scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  --shared-root "${ROUTEA_ROOT}" \
  --route-a-root "${ROUTEA_LABEL_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 1
```

Expected:

```bash
test -f "${ROUTEA_ROOT}/sub-2001/reference/kong2022/reference_summary.json"
test -f "${ROUTEA_ROOT}/sub-2001/reference/kong2022/stage_manifest.json"
```

### E6. RouteB ROI FC smoke

```bash
"${PY314}" scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --reference-root "${ROUTEB_ROOT}" \
  --mask-root "${SURFACE_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --left-cortex-surface-template "${LEFT_SURF_TEMPLATE}" \
  --right-cortex-surface-template "${RIGHT_SURF_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 1
```

Expected: FC files and stage manifests under:

```text
${ROUTEB_ROOT}/sub-2001/fc/
```

Use bounded checks:

```bash
timeout 60 find "${ROUTEB_ROOT}/sub-2001" -maxdepth 3 2>/dev/null | head -80
```

### E7. Intrinsic FC smoke

```bash
"${PY314}" scripts/core_workflow/04_02_materialize_intrinsic_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 1
```

Expected: intrinsic FC artifacts under `${ROUTEB_ROOT}/sub-2001/`.

### E8. Smoke decision gate

Before scaling to all subjects, confirm:

- `02_01` works with the Stanford `.nii.gz` BOLD template, or a documented uncompressed-output fallback exists under `outputs_stanford/`.
- `03_01` and `03_02` read Stanford dtseries through `--dtseries-template` without symlink tree.
- `04_01` reads Stanford cortex surfaces through existing surface templates.
- RouteA labels satisfy the flat label contract.
- All smoke stage manifests exist and record the Stanford source paths.

If any item fails, fix the code/template/documentation and rerun only the failed smoke step.

## Phase F: Full 199-subject execution

Only start this phase after smoke success. Keep all concurrency at or below the table below.

| Step | Jobs | Notes |
|---|---:|---|
| `01_01` HippUnfold | 4 external subjects | HippUnfold script should use internal `--cores 1`; do not exceed 4 concurrent subjects |
| `02_01` prepare inputs | 2 for Workbench-heavy stages | Script hard-caps Workbench subject jobs; pass `--jobs 2` |
| `02_02` manifest | 1 | Fast |
| `03_01` / `03_02` references | 4 | Thread pool |
| `04_01` / `04_02` FC | 4 | Process pool |
| `05_01` gradients | 1 | Compute/memory heavy |
| `05_02` atlas | 1 | Compute/memory heavy |
| `06_xx` / `07_01` rendering | 1 | Final products |

### F1. Build full subject list

Use an existing validated subject list if available. If no list exists, create one with bounded commands and review it:

```bash
timeout 60 find "${STANFORD_ROOT}" -maxdepth 1 -type d -name "sub-*" 2>/dev/null | sort > "${KONG_ROOT}/manifests/stanford_all_subject_dirs.txt"
wc -l "${KONG_ROOT}/manifests/stanford_all_subject_dirs.txt"
head -5 "${KONG_ROOT}/manifests/stanford_all_subject_dirs.txt"
tail -5 "${KONG_ROOT}/manifests/stanford_all_subject_dirs.txt"
```

Convert absolute paths to workflow subject IDs before using them as a manifest. Use `sub-XXXX` form because downstream scripts strip `sub-` when they need bare IDs:

```bash
sed 's#.*/##' "${KONG_ROOT}/manifests/stanford_all_subject_dirs.txt" > "${KONG_ROOT}/manifests/stanford_all_subjects.txt"
wc -l "${KONG_ROOT}/manifests/stanford_all_subjects.txt"
head -5 "${KONG_ROOT}/manifests/stanford_all_subjects.txt"
tail -5 "${KONG_ROOT}/manifests/stanford_all_subjects.txt"
```

Expected: one subject per line, e.g. `sub-2001`, with no full paths. If the count is not 199, stop and determine whether Stanford has missing subjects or the expected cohort size was wrong.

### F2. Run HippUnfold for all subjects

Do not launch more than 4 simultaneous HippUnfold subject jobs. Use the repository's existing launcher if one already exists and obeys the hard cap. If writing a new launcher, document it and keep concurrency configurable with default 4.

Per-subject command template:

```bash
"${PY314}" scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  "${SUBJECT_ID}" \
  "${STANFORD_ROOT}" \
  "${HIPPO_BATCH}/sub-${SUBJECT_ID}/hippunfold"
```

After batch completion, check a few outputs with bounded listing:

```bash
timeout 60 find "${HIPPO_BATCH}" -maxdepth 2 2>/dev/null | head -80
```

### F3. Check HippUnfold outputs

```bash
"${PY314}" scripts/core_workflow/01_02_check_hippunfold_outputs.py \
  --dataset stanford \
  --input-root "${STANFORD_ROOT}"
```

If `01_02` assumes HCP paths or output roots, update it with explicit Stanford root/output args and update docs in the same change. Do not infer success from directory count alone; verify QC outputs and 512-density surface assets.

### F4. Prepare group inputs

```bash
"${PY314}" scripts/core_workflow/02_01_prepare_group_inputs.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --route-root "${KONG_ROOT}" \
  --shared-root "${SURFACE_ROOT}" \
  --template-root "${TEMPLATE_ROOT}" \
  --bold-root "${STANFORD_ROOT}" \
  --bold-template "${BOLD_TEMPLATE}" \
  --jobs 2
```

If subject discovery misses subjects because `load_subjects(...)` still intersects with legacy `bold_root/*`, pass an explicit full subject manifest:

```bash
"${PY314}" scripts/core_workflow/02_01_prepare_group_inputs.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --route-root "${KONG_ROOT}" \
  --shared-root "${SURFACE_ROOT}" \
  --template-root "${TEMPLATE_ROOT}" \
  --bold-root "${STANFORD_ROOT}" \
  --bold-template "${BOLD_TEMPLATE}" \
  --subjects-manifest "${KONG_ROOT}/manifests/stanford_all_subjects.txt" \
  --jobs 2
```

### F5. Build QC-pass manifest

```bash
"${PY314}" scripts/core_workflow/02_02_build_qc_manifest.py \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --output-manifest "${MANIFEST}"
```

Check:

```bash
wc -l "${MANIFEST}"
head -5 "${MANIFEST}"
tail -5 "${MANIFEST}"
```

Record the QC-pass subject count before continuing.

### F6. Build RouteB cortex reference

```bash
"${PY314}" scripts/core_workflow/03_01_build_kong2022_reference.py \
  --shared-root "${ROUTEB_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 4
```

### F7. Build RouteA cortex reference

```bash
"${PY314}" scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  --shared-root "${ROUTEA_ROOT}" \
  --route-a-root "${ROUTEA_LABEL_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --kong2022-dlabel "${KONG2022_DLABEL}" \
  --jobs 4
```

If this fails because of RouteA labels, fix only the label bridge/conversion. Do not replace the dtseries/template approach with a full symlink tree.

### F8. Materialize RouteB FC

```bash
"${PY314}" scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --reference-root "${ROUTEB_ROOT}" \
  --mask-root "${SURFACE_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --left-cortex-surface-template "${LEFT_SURF_TEMPLATE}" \
  --right-cortex-surface-template "${RIGHT_SURF_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

### F9. Materialize RouteA FC

RouteA FC should read RouteA references and the same Stanford dtseries/surface templates:

```bash
"${PY314}" scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  --shared-root "${ROUTEA_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --reference-root "${ROUTEA_ROOT}" \
  --mask-root "${SURFACE_ROOT}" \
  --route-a-root "${ROUTEA_LABEL_ROOT}" \
  --data-root "${STANFORD_ROOT}" \
  --dtseries-template "${DT_TEMPLATE}" \
  --hippunfold-batch-root "${HIPPO_BATCH}" \
  --left-cortex-surface-template "${LEFT_SURF_TEMPLATE}" \
  --right-cortex-surface-template "${RIGHT_SURF_TEMPLATE}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

If `04_01` needs an explicit route/branch flag to distinguish RouteA vs RouteB in the current code, inspect the parser and pass that flag. Do not assume based only on root names.

### F10. Materialize intrinsic FC

RouteB:

```bash
"${PY314}" scripts/core_workflow/04_02_materialize_intrinsic_fc.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

RouteA, if needed by downstream branch:

```bash
"${PY314}" scripts/core_workflow/04_02_materialize_intrinsic_fc.py \
  --shared-root "${ROUTEA_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --subjects-manifest "${MANIFEST}" \
  --jobs 4
```

### F11. Compute gradients

RouteB:

```bash
"${PY314}" scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

RouteA, if needed:

```bash
"${PY314}" scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  --shared-root "${ROUTEA_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

### F12. Run group atlas

RouteB:

```bash
"${PY314}" scripts/core_workflow/05_02_run_group_atlas.py \
  --dataset stanford \
  --shared-root "${ROUTEB_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

RouteA, if needed:

```bash
"${PY314}" scripts/core_workflow/05_02_run_group_atlas.py \
  --dataset stanford \
  --shared-root "${ROUTEA_ROOT}" \
  --surface-root "${SURFACE_ROOT}" \
  --jobs 1
```

If the script does not accept `--dataset stanford` or infers HCP output roots, update it explicitly and update docs in the same change.

### F13. Render final products

Commands may need additional `--group-root`, `--out-root`, `--shared-root`, or route flags depending on existing parser contracts. Inspect each parser before running. Do not rely on HCP defaults.

Baseline:

```bash
"${PY314}" scripts/core_workflow/06_01_build_group_overview.py \
  --dataset stanford

"${PY314}" scripts/core_workflow/07_01_build_group_matrix.py \
  --dataset stanford
```

If defaults point to `outputs_hcp`, rerun with explicit Stanford roots and update this plan/README with the exact working command.

## Final verification checklist

For each completed stage, verify:

- Stage-specific `stage_manifest.json` or equivalent summary exists.
- Manifest records Stanford source paths, not `data/stanford_input` fake HCP paths.
- FC/gradient/group-atlas outputs are under `outputs_stanford/`, not `outputs_hcp/`.
- Consumed hippocampal assets show `den-512` in actual paths where required.
- Subject count in downstream products matches the QC-pass manifest count.
- RouteA and RouteB outputs are not accidentally written into each other's roots.
- All commands used `/home/yojiao/.conda/envs/py314/bin/python` except direct HippUnfold CLI validation under `hippo2`.

Useful bounded checks:

```bash
wc -l "${MANIFEST}"
timeout 60 find "${SURFACE_ROOT}" -maxdepth 2 2>/dev/null | head -80
timeout 60 find "${ROUTEB_ROOT}" -maxdepth 3 2>/dev/null | head -80
timeout 60 find "${ROUTEA_ROOT}" -maxdepth 3 2>/dev/null | head -80
timeout 60 find "${STANFORD_OUT}/work" -maxdepth 3 2>/dev/null | head -80
```

If any command times out with exit code 124, report it and stop. Do not retry unbounded.

## Known open items to resolve during execution

1. **Kong2022 dlabel path**: must be found and passed as `--kong2022-dlabel` for `03_01`/`03_02` if required. Do not guess.
2. **RouteA label layout**: verify `ArealMSHBM_Kong2022/` contract. Only bridge labels if necessary.
3. **`.nii.gz` BOLD compatibility**: smoke `02_01` determines whether Workbench accepts Stanford compressed BOLD directly. If not, use an `outputs_stanford/tmp_bold/` decompression cache, not source-data writes.
4. **Later-stage Stanford root defaults**: `05_xx`, `06_xx`, and `07_01` may still default to HCP output roots. Inspect parser and pass explicit roots or patch with Stanford-safe args before full rendering.
5. **Doc sync after any code change**: update `scripts/core_workflow/README.md` and this plan if actual commands differ from the commands above.

# Iowa7T Branch Downstream Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fixed-layout Iowa7T branch downstream adapter for completed ArealMSHBM subject/echo branches.

**Architecture:** Add one Iowa7T-specific adapter module that resolves branch paths, validates inputs, and calls/copies only low-level core workflow operations that can write directly into a branch root. Keep existing HCP-style numbered scripts unchanged because their root interface appends `sub-<ID>`.

**Tech Stack:** Python 3.14 repo runtime at `/home/yojiao/.conda/envs/py314/bin/python`, nibabel/numpy/scipy, Connectome Workbench wrapper, pytest.

## Global Constraints

- Use local Jianglab prefix `/home/yojiao/LSS/jianglab` on this system.
- Output root is `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`.
- Do not create `variants/`, empty `group/`, or nested branch compatibility trees such as `<branch_root>/sub-<ID>/`.
- `cortex/common/` is a real non-empty branch output for cortex tSNR masks and run-boundary summaries.
- Kong2022 ROI outputs preserve 400 rows/columns; missing individualized label parcels are represented with NaN rows/columns and explicit metadata.
- Maximum concurrency is 10; default adapter execution is sequential.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python commands.
- Do not delete files or directories.

---

### Task 1: Path contract tests

**Files:**
- Create: `tests/iowa7t/test_run_branch_downstream.py`
- Create: `scripts/iowa7t/run_branch_downstream.py`

**Interfaces:**
- Produces: `BranchJob`, `resolve_branch_job(subject: str, branch: str, *, data_root: Path, output_root: Path) -> BranchJob`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from scripts.iowa7t.run_branch_downstream import resolve_branch_job


def test_resolve_branch_job_writes_direct_branch_paths(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multiband", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    assert job.branch_root == tmp_path / "hipp_outputs" / "subjects" / "sub-06112026" / "multiband"
    assert job.surface_root == job.branch_root / "surface"
    assert job.cortex_kong2022_root == job.branch_root / "cortex" / "kong2022"
    assert job.fc_root == job.branch_root / "fc" / "kong2022"
    assert job.areal_mshbm_label_dir == job.branch_root / "cortex" / "areal_mshbm"
    assert "multiband/sub-06112026" not in str(job.surface_root)


def test_resolve_branch_job_uses_branch_specific_input_templates(tmp_path: Path) -> None:
    single = resolve_branch_job("06112026", "multiband", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")
    multi = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    assert single.bold_volume.name == "sub-06112026_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
    assert single.dtseries.name == "sub-06112026_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    assert multi.bold_volume.name == "sub-06112026_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-MEICA_XCPD36P_denoised_bold.nii.gz"
    assert multi.dtseries.name == "sub-06112026_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/iowa7t/test_run_branch_downstream.py -q`

Expected: FAIL because `scripts.iowa7t.run_branch_downstream` does not exist or lacks the interface.

- [ ] **Step 3: Implement minimal resolver**

Create `BranchJob` and `resolve_branch_job(...)` with the exact path rules in the spec.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/iowa7t/test_run_branch_downstream.py -q`

Expected: PASS.

### Task 2: Direct branch stage implementation

**Files:**
- Modify: `scripts/iowa7t/run_branch_downstream.py`

**Interfaces:**
- Produces CLI: `--job <subject>:<branch>`, `--stage all|surface|reference|fc`, `--overwrite`, `--jobs`

- [ ] **Step 1: Add direct stage functions**

Implement direct branch versions of surface sampling/tSNR/smoothing, reference extraction, and FC materialization. Functions write only inside `BranchJob.branch_root`.

- [ ] **Step 2: Validate syntax and tests**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/iowa7t/run_branch_downstream.py
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/iowa7t/test_run_branch_downstream.py -q
```

Expected: both commands exit 0.

### Task 3: Run three completed ArealMSHBM branches

**Files:**
- Data outputs only under fixed branch roots in Jianglab storage.

**Interfaces:**
- Consumes completed labels:
  - `sub-06112026/multiband/cortex/areal_mshbm/`
  - `sub-06122026/multiband/cortex/areal_mshbm/`
  - `sub-06112026/multi_echo/cortex/areal_mshbm/`

- [ ] **Step 1: Run adapter sequentially**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/run_branch_downstream.py \
  --job 06112026:multiband \
  --job 06122026:multiband \
  --job 06112026:multi_echo \
  --stage all \
  --jobs 1
```

- [ ] **Step 2: Verify outputs**

Run a py314 verification command that checks each branch for:

- `surface/raw/sub-<ID>_hemi-L_space-corobl_den-512_label-hipp_bold.func.gii`
- `surface/2mm/sub-<ID>_hemi-L_timeseries.npy`
- `surface/4mm/sub-<ID>_hemi-R_timeseries.npy`
- `cortex/kong2022/cortex_roi_parcel_timeseries.npy`
- `cortex/common/runs/run-1/reference_summary.json`
- `fc/kong2022/2mm/hemi_L/hipp_vertex_to_network_fc.npy`
- `fc/kong2022/4mm/hemi_R/fc_summary.json`

Expected: all files exist for all three jobs.

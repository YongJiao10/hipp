# ESTT Kong2022 Shifted-Cosine Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the ESTT 25-subject Kong2022 group atlas with the repository's formal shifted-cosine affinity and restore the missing 400-ROI cortex timeseries for sub-677 and sub-778.

**Architecture:** Reuse the existing ESTT subject-level extraction and FC scripts for the two missing subjects, then run a small repository-owned group refresh entrypoint that consumes all 25 subject Fisher-z vertex-to-ROI matrices, computes group HippoMaps gradients, applies shifted-cosine surface-constrained spectral clustering, and writes the atlas into the formal ESTT `group_atlases/group-gradient-spectral_kong2022_8net_{2mm,4mm}` directories. The manifest and directory name both record the authoritative method as `gradient-spectral` with `surface_affinity: shifted-cosine`.

**Tech Stack:** `/home/yojiao/.conda/envs/py314/bin/python`, NumPy, SciPy, scikit-learn, nibabel/GIFTI, BrainSpace through the repository HippoMaps adapter, and the existing ESTT Kong2022 extraction/FC scripts.

## Global Constraints

- Use only the ESTT 25-subject manifest: `subjects_25_estt_kong2022_areal.txt`.
- The two restored cortex ROI caches must be exactly `(400, n_timepoints)` and use the existing Kong2022 dlabel and `TSNR >= 25` gate.
- The group atlas must consume the recorded HippUnfold `den-512` corobl midthickness surface topology and use 2mm/4mm separately for both L/R hemispheres.
- The formal gradient affinity is `((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0`; do not label a ReLU-cosine result as shifted-cosine.
- Do not access `_archive` content and do not delete existing files; overwrite only the requested Kong2022 canonical outputs after successful computation.
- Keep process concurrency at or below 10 and use the explicit py314 interpreter for repository Python tasks.

---

### Task 1: Add failing tests for ESTT atlas refresh helpers

**Files:**
- Create: `tests/core_workflow/test_estt_kong2022_group_atlas.py`
- Test: `tests/core_workflow/test_estt_kong2022_group_atlas.py`

**Interfaces:**
- Consumes: The planned pure helpers `validate_fc_matrix`, `cluster_to_functional_labels`, and `atlas_provenance` from `scripts/core_workflow/refresh_estt_kong2022_group_atlas.py`.
- Produces: Regression tests proving 512x400 validation, canonical8 cluster mapping, and shifted-cosine provenance.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from scripts.core_workflow.refresh_estt_kong2022_group_atlas import (
    atlas_provenance,
    cluster_to_functional_labels,
    validate_fc_matrix,
)


def test_validate_fc_matrix_rejects_non_400_roi_input():
    with pytest.raises(ValueError, match="512x400"):
        validate_fc_matrix(np.zeros((512, 399), dtype=np.float32), "sub-677")


def test_cluster_to_functional_labels_uses_dominant_canonical_network():
    labels = np.array([1, 1, 2, 2, 0], dtype=np.int32)
    profile = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=np.float32)
    out, mapping = cluster_to_functional_labels(
        labels,
        profile,
        ["Default", "Visual"],
    )
    assert out.tolist() == [1, 1, 2, 2, 0]
    assert mapping == {"1": 1, "2": 2}


def test_atlas_provenance_records_shifted_cosine_and_estt_cohort():
    meta = atlas_provenance(subjects=["369", "677"], smoothing="2mm", hemi="L", k=3)
    assert meta["surface_affinity"] == "shifted-cosine"
    assert meta["affinity_description"] == "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0"
    assert meta["n_subjects"] == 2
```

- [ ] **Step 2: Run the focused test to verify the expected missing-module failure**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_estt_kong2022_group_atlas.py -q
```

Expected: FAIL because `refresh_estt_kong2022_group_atlas.py` does not exist yet.

### Task 2: Implement the ESTT shifted-cosine group-atlas refresh entrypoint

**Files:**
- Create: `scripts/core_workflow/refresh_estt_kong2022_group_atlas.py`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md` (add the ESTT refresh entrypoint and output contract)
- Modify: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md` (record the ESTT 25-subject refresh lineage)
- Test: `tests/core_workflow/test_estt_kong2022_group_atlas.py`

**Interfaces:**
- Consumes: `shared/sub-*/fc/kong2022/{2mm,4mm}/hemi_{L,R}/vertex_to_roi_fisherz.npy`, `dense_corobl_batch_t1only/sub-*/hippunfold/sub-*/surf/*den-512_label-hipp_midthickness.surf.gii`, the 25-subject manifest, and the repository shifted-cosine/BrainSpace adapters.
- Produces: Per-smoothing/per-hemi group labels, functional labels, label maps, K summaries, group mean FC/gradient arrays, and manifests under the requested ESTT `group_atlases` compatibility directories.

- [ ] **Step 1: Add the smallest implementation that satisfies the failing tests**

Implement these exact behaviors:

```python
def validate_fc_matrix(matrix: np.ndarray, subject: str) -> np.ndarray:
    # Return float32 matrix; require exactly 512 vertices and 400 ROIs.

def cluster_to_functional_labels(
    cluster_labels: np.ndarray,
    cluster_roi_profile: np.ndarray,
    canonical8_order: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    # Map each cluster to the highest-scoring canonical8 profile; preserve 0 as null.

def atlas_provenance(*, subjects: list[str], smoothing: str, hemi: str, k: int) -> dict[str, object]:
    # Record ESTT cohort size and the exact shifted-cosine definition.
```

Then implement the CLI with defaults pointing to:

```text
data root:       /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P
shared root:     <data root>/hipp_outputs/estt_kong2022_areal_individual_spectral/shared
surface root:    <data root>/hipp_outputs/dense_corobl_batch_t1only
subjects:        <...>/hipp_outputs/estt_kong2022_areal_individual_spectral/manifests/subjects_25_estt_kong2022_areal.txt
atlas root:      <data root>/hipp_outputs/group_atlases
```

For each of `2mm/4mm × L/R`, load all 25 `(512,400)` Fisher-z matrices, use finite-row coverage `>=0.80`, average across subjects, fit five BrainSpace gradients with the existing adapter, select G1-G3, build the existing `build_gradient_spectral_affinity`, run fixed primary K=3 spectral clustering with `n_init=50`, reorder labels along the template midthickness long axis, and write both cluster and canonical8 functional labels. Write paths into manifests as absolute paths under the ESTT target; set `method` to `gradient-spectral`, `atlas` to `kong2022`, `n_subjects` to 25, and `surface_affinity` to `shifted-cosine`.

- [ ] **Step 2: Run the focused tests to verify they pass**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_estt_kong2022_group_atlas.py -q
```

Expected: PASS.

### Task 3: Restore the two missing Kong2022 cortex ROI timeseries and FC inputs

**Files:**
- Modify external output: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/shared/sub-677/reference/kong2022/`
- Modify external output: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/shared/sub-778/reference/kong2022/`
- Modify external output: the corresponding `fc/kong2022/{2mm,4mm}/hemi_{L,R}/` directories for sub-677 and sub-778

**Interfaces:**
- Consumes: existing ESTT extraction script, subject-specific ArealMSHBM label GIFTIs, source dtseries, and existing 512-vertex GIFTI hippocampal timeseries.
- Produces: `cortex_roi_parcel_timeseries.npy`, `cortex_roi_parcels.tsv`, `reference_summary.json`, `stage_manifest.json`, and eight `(512,400)` FC matrices plus summaries for the two subjects.

- [ ] **Step 1: Run the existing extractor for only sub-677 and sub-778**

```bash
/home/yojiao/.conda/envs/py314/bin/python /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/scripts/extract_cortex_kong2022_areal_mshbm_roi_timeseries.py --subject 677 --shared-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/shared --labels-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm --overwrite
/home/yojiao/.conda/envs/py314/bin/python /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/scripts/extract_cortex_kong2022_areal_mshbm_roi_timeseries.py --subject 778 --shared-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/estt_kong2022_areal_individual_spectral/shared --labels-root /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm --overwrite
```

- [ ] **Step 2: Run the existing FC materializer for the 8 subject/hemi/smoothing combinations**

Use `materialize_vertex_to_roi_fisherz.py` with `--shared-root` set to the ESTT shared root and `--hipp-shared-root` set to `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/hipp_functional_parcellation_network/shared`; keep concurrency at four or fewer.

- [ ] **Step 3: Verify the two restored ROI matrices and FC matrices**

Run a py314 validation that checks each ROI matrix has 400 rows, each FC matrix has shape `(512,400)`, all eight summaries report aligned timepoints, and the 512-vertex source GIFTIs exist for both subjects.

### Task 4: Compute and overwrite the ESTT Kong2022 group atlas

**Files:**
- Modify external output: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/group_atlases/group-gradient-spectral_kong2022_8net_2mm/`
- Modify external output: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/group_atlases/group-gradient-spectral_kong2022_8net_4mm/`

**Interfaces:**
- Consumes: The verified 25-subject FC bank, the 512-vertex corobl midthickness surface topology, and the refresh entrypoint from Task 2.
- Produces: Four refreshed L/R atlas outputs (2mm and 4mm), each with 25-subject and shifted-cosine provenance. Existing Kong2019 directories remain untouched.

- [ ] **Step 1: Run the refresh entrypoint over `2mm 4mm` and `L R`**

Run the explicit py314 script with `--force` and the ESTT 25-subject manifest. The command must finish all four combinations before any “updated” claim is made.

- [ ] **Step 2: Verify the refreshed manifests and labels**

Check every refreshed manifest has `n_subjects == 25`, `surface_affinity == "shifted-cosine"`, the exact shifted-cosine description, `method == "gradient-spectral"`, and local absolute output paths. Check every label vector has length 512 and every functional label uses only `0..8` plus the documented unassigned key when present.

### Task 5: Final end-to-end verification and handoff

**Files:**
- Verify only; no additional source edits.

- [ ] **Step 1: Confirm the subject inventory is complete**

Confirm the ESTT Kong2022 reference bank and FC bank contain all 25 IDs, including sub-677 and sub-778, and that the hippocampal 512-vertex GIFTIs remain present for all 25.

- [ ] **Step 2: Confirm no old HCP provenance remains in the refreshed Kong2022 manifests**

Search the two refreshed Kong2022 directories for `outputs_experiments`, `n_subjects": 151`, `relu_cosine`, and `max(cosine_similarity,0)`; each search must return no active manifest match.

- [ ] **Step 3: Run the focused repository tests and report exact output paths**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_estt_kong2022_group_atlas.py -q
```

Report the two restored ROI timeseries paths, the two refreshed atlas directory paths, subject count, shapes, and affinity provenance.

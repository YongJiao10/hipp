# JJ_TMS Site-to-Kong2022 Network Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible subject-level table assigning each available JJ_TMS stimulation coordinate to the subject's individualized Kong2022 parcel, 17-network family, and merged 8-network family, with geometric and neighborhood QC.

**Architecture:** A standalone core-workflow CLI reads only the required fields from the supplied XLSX with Python's standard-library ZIP/XML readers, intersects those records with the fixed 70-subject manifest, and projects each native-space target to the nearest vertex on that subject's fsLR-32k midthickness surface. It joins the selected individualized parcel ID to the repository's canonical Kong2022 registry and writes CSV plus JSON summary/provenance under the JJ_TMS present tree.

**Tech Stack:** Python 3.14, NumPy, SciPy `cKDTree`, NiBabel, pytest, existing `lib.kong2022_label_utils`.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for all Python execution.
- Do not use bare `python`.
- Do not access `_archive` or any archive directory.
- Maximum concurrent processes is 10; this implementation is single-process.
- Do not delete any file or directory.
- The individualized labels remain the existing `lh_labels.txt` and `rh_labels.txt`; do not alter Kong2022_ArealMSHBM model logic.
- Native coordinates are authoritative for projection; preserve MNI coordinates as input provenance and do not infer native geometry from MNI arguments alone.
- Code behavior, paths, and workflow outputs require matching documentation in the same change.

---

### Task 1: Pure coordinate, label, and workbook contracts

**Files:**
- Create: `tests/core_workflow/test_jj_tms_target_network_mapping.py`
- Create: `scripts/core_workflow/map_jj_tms_sites_to_kong2022.py`

**Interfaces:**
- Consumes: `read_xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str | None]]`; `project_target_to_surface(target_xyz: np.ndarray, surface_xyz: np.ndarray, parcel_labels: np.ndarray, roi_by_id: dict[int, dict], neighborhood_radius_mm: float) -> dict[str, object]`.
- Produces: validated workbook records and one deterministic projection row containing vertex, distance, parcel/network, and 5-mm neighborhood QC.

- [ ] **Step 1: Write failing tests with hand-derived fixtures**

```python
def test_projection_chooses_nearest_vertex_and_reports_network_qc() -> None:
    surface = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0]], dtype=float)
    labels = np.array([11, 12, 12], dtype=np.int32)
    registry = {
        11: {"roi_id": 11, "family17": "DefaultA", "canonical8": "Default"},
        12: {"roi_id": 12, "family17": "ContA", "canonical8": "Control"},
    }
    result = module.project_target_to_surface(
        np.array([1.8, 0.1, 0]), surface, labels, registry, 1.5
    )
    assert result["vertex_index"] == 1
    assert result["parcel_id"] == 12
    assert result["family17"] == "ContA"
    assert result["neighborhood_vertex_count"] == 2
    assert result["neighborhood_primary_family17_fraction"] == 1.0
```

Add independent tests for XLSX shared strings/inline strings, length mismatch, unknown/zero parcel IDs, and an empty radius neighborhood fallback to the nearest vertex.

- [ ] **Step 2: Run tests and verify RED**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_jj_tms_target_network_mapping.py -q`

Expected: collection fails because `scripts/core_workflow/map_jj_tms_sites_to_kong2022.py` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

Implement XLSX sheet lookup through `xl/workbook.xml` relationships, sparse-cell reference decoding, coordinate validation, hemisphere validation, nearest-vertex lookup, and deterministic neighborhood counts. Reject missing coordinate components, surface/label length mismatch, parcel 0, unknown parcel IDs, and hemisphere/parcel registry disagreement with clear `ValueError`s.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_jj_tms_target_network_mapping.py -q`

Expected: all Task 1 tests pass without warnings.

### Task 2: Cohort CLI and provenance outputs

**Files:**
- Modify: `tests/core_workflow/test_jj_tms_target_network_mapping.py`
- Modify: `scripts/core_workflow/map_jj_tms_sites_to_kong2022.py`

**Interfaces:**
- Consumes: Task 1 pure functions, `manifests/jj_tms_hippunfold_subjects.txt`, the provided workbook, local JJ_TMS surface root, individualized label root, and `load_kong2022_bundle()`.
- Produces: `map_cohort(...) -> tuple[list[dict[str, object]], dict[str, object]]` and a CLI writing `jj_tms_kong2022_tms_site_networks.csv` plus `jj_tms_kong2022_tms_site_networks_summary.json`.

- [ ] **Step 1: Write failing cohort tests**

Create a two-subject temporary fixture with real GIFTI surfaces, text labels, a minimal XLSX, and a two-line manifest. Assert exact subject order, hemisphere selection, CSV columns, summary counts, input path provenance, and failure when a manifest subject is missing from the workbook.

- [ ] **Step 2: Run tests and verify RED**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_jj_tms_target_network_mapping.py -q`

Expected: cohort/CLI tests fail because `map_cohort` and output writers are absent.

- [ ] **Step 3: Implement minimal cohort orchestration**

Normalize `JJ_TMS_1001` and `sub-1001` to `1001`, preserve the fixed manifest order, require exactly one coordinate record per requested subject, select hemisphere from `visit2_target` with native-x agreement, load the matching 32k midthickness and individualized label files, and write atomically through temporary sibling files followed by `Path.replace`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_jj_tms_target_network_mapping.py -q`

Expected: all tests pass without warnings.

### Task 3: Run the 70-subject analysis, document it, and verify

**Files:**
- Create: `docs/hipp_parcellation_network/jj_tms_target_network_mapping.md`
- Create at runtime: `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_tms_site_networks.csv`
- Create at runtime: `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_tms_site_networks_summary.json`

**Interfaces:**
- Consumes: Task 2 CLI with canonical defaults.
- Produces: the requested 70-row result table and reproducibility/QC documentation.

- [ ] **Step 1: Run the canonical cohort command**

Run: `/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/map_jj_tms_sites_to_kong2022.py`

Expected: 70 output rows; 68 mapped subjects; explicit unmapped rows for `sub-3003` (no workbook record) and `sub-3026` (native coordinates absent); valid parcel IDs for mapped rows; and a JSON summary containing family distributions and nearest-surface distance statistics.

- [ ] **Step 2: Write synchronized documentation**

Document the coordinate-space decision, canonical input/output paths, exact command, CSV schema, 17-network versus merged 8-network interpretation, nearest-vertex primary rule, 5-mm Euclidean neighborhood QC, and cautions for boundary cases or unusually large target-to-surface distances.

- [ ] **Step 3: Run focused and contract verification**

Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_jj_tms_target_network_mapping.py tests/core_workflow/test_routea_flat_label_contract.py -q`

Run: `/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/map_jj_tms_sites_to_kong2022.py`

Expected: all tests pass and compilation exits 0.

- [ ] **Step 4: Audit result invariants**

Read the generated JSON and assert: `n_requested_subjects == 70`, `n_output_rows == 70`, `n_mapped_subjects == 68`, `n_unmapped_subjects == 2`, all subject IDs are unique, mapped nearest distances are finite/nonnegative, and neighborhood fractions lie in `[0, 1]`. Preserve `sub-3003` as `missing_workbook_record` and `sub-3026` as `missing_native_coordinates`. Report any subjects with distance above 5 mm or neighborhood primary-network fraction below 0.5 as explicit QC flags rather than silently reassigning them.

No commit or deletion is part of this plan; the user did not request either operation.

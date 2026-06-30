# Core Workflow Code Audit: Obsolete, Unused, and Redundant Code (RESOLVED)

This audit document summarizes the analysis of the `scripts/core_workflow/` codebase, identifying obsolete scripts, unused library modules, duplicate/redundant functions, and library symbols. 

> [!NOTE]
> All identified redundancies and numbering registration issues have been successfully resolved and refactored. The test suite passes 100%.

---

## 1. Registered Scripts (RESOLVED)

### ⚠️ Deprecated Script
*   **[scripts/core_workflow/06_05_render_group_gradient_heatmaps.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/06_05_render_group_gradient_heatmaps.py)**
    *   **Status:** **DEPRECATED & RETIRED**
    *   **Details:** The script's docstring states: *"DEPRECATED: Locked-scene composite heatmap output is retired."* Running the script immediately prints a deprecation warning JSON and exits with code `2`.
    *   **Replacement:** Use `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py` outputs instead.

### 🚫 Numbered Scripts Contract Test Failures
*   **Status:** **RESOLVED**
*   **Action taken:** Registered both scripts in [step_registry.json](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/step_registry.json) and [tests/core_workflow/test_pipeline_numbering_contract.py](file:///home/yojiao/hpchome/hipp/tests/core_workflow/test_pipeline_numbering_contract.py).
*   **Resolved Scripts:**
    1.  **[scripts/core_workflow/05_05_build_weighted_fisherz_branch.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/05_05_build_weighted_fisherz_branch.py)**
    2.  **[scripts/core_workflow/06_08_render_hcp_subfield_composition.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/06_08_render_hcp_subfield_composition.py)**

---

## 2. Unused Library Modules & Layout Functions (ARCHIVED)

### 📦 Completely Unused Library Module
*   **Status:** **RESOLVED (ARCHIVED)**
*   **Action taken:** Archived both files to the `_archive/` directory.
*   **Resolved Files:**
    1.  `scripts/core_workflow/lib/iowa_hcp500.py` -> **[_archive/iowa_hcp500.py](file:///home/yojiao/hpchome/hipp/_archive/iowa_hcp500.py)**
    2.  `tests/core_workflow/test_iowa_hcp500_workflow.py` -> **[_archive/test_iowa_hcp500_workflow.py](file:///home/yojiao/hpchome/hipp/_archive/test_iowa_hcp500_workflow.py)**

### 📂 Unused Iowa7T Dataset Layout Functions
*   **[scripts/core_workflow/lib/output_layout.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/output_layout.py)**
    *   **Status:** Retained (Only used in `test_iowa7t_36p_layout.py` and layout verification).

---

## 3. Duplicate and Redundant Code (RESOLVED & MERGED)

All redundancies have been merged and cleaned up:

### 🔄 ROI Extraction Helpers
*   `write_tsv`
    *   **Action taken:** Removed local definitions in `lib/extract_cortex_kong2022_roi_timeseries.py` and `lib/extract_cortex_kong2022_route_a_roi_timeseries.py`. Imported and shared standard `write_tsv` from **[lib/io_utils.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/io_utils.py)**.
*   `compute_cortex_tsnr_gate` & `parse_run_segments`
    *   **Action taken:** Extracted and consolidated both functions into **[lib/io_utils.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/io_utils.py)**. Imported them back in both `extract_cortex_kong2022_roi_timeseries.py` and `extract_cortex_kong2022_route_a_roi_timeseries.py`.

### 🎨 Network Color Loader Wrappers
*   `load_network_colors`
    *   **Action taken:** Deleted the wrapper function inside **[lib/surface_utils.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/surface_utils.py)**. Directly imported and called `load_network_colors` from **[lib/network_colors.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/network_colors.py)**.

### 👁️ Scene Rendering Helpers
*   `capture_scene`
    *   **Action taken:** Consolidated into the dedicated Connectome Workbench interface module **[lib/workbench.py](file:///home/yojiao/hpchome/hipp/scripts/core_workflow/lib/workbench.py)**. Removed duplicate definitions in `lib/render_cortex_pfm_scene.py` and `lib/workbench_scene_batch.py`, importing from `lib.workbench` instead.

# Area Summary Layout and HCP 7T Cross-Fit Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the current PNG-only Kember area summary readable after adding the HCP 7T reference, and add a scientifically valid HCP 7T-versus-NTHC comparison whose HCP labels come from the family-aware held-out cross-fit templates.

**Architecture:** Keep the existing Stanford direct-contrast result table unchanged and add HCP 7T as a separate descriptive reference branch. The HCP branch will validate the maintained five-fold cross-fit manifests, load each held-out subject's `individual_labels_k3.npy`, transfer those 512-source labels to the HCP native surface with the existing `ADAP_BARY_AREA` helper, and sum native `surfareacorobl` values with the existing Kember area function. The plotter will combine the resulting HCP rows with Stanford rows only at render time, compute an explicit HCP 7T minus NTHC Welch effect, and render one PNG with explicit spacing constants.

**Tech Stack:** Python 3.14, pandas, NumPy, SciPy, nibabel, Matplotlib, pytest, Connectome Workbench.

**Spec:** User request in the current conversation; there is no separate specification file.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python tasks.
- HCP 7T template matching must use the maintained family-aware five-fold cross-fit subject labels; never use the full-concat HCP template labels for an HCP subject's own area.
- Kember area remains native HippUnfold `surfareacorobl` summed over native system labels.
- The HCP 7T comparison is a descriptive cross-dataset reference and must not be silently appended to the Stanford-only direct-contrast result table or its FDR family.
- The current entry point writes PNG only and must not create or overwrite PDF files.
- Existing PDF figure artifacts are removed only through the current-session exact-path confirmation; the maintained output is PNG-only.
- Update code, configuration, tests, provenance, and user-facing documentation together.

---

### Task 1: Lock HCP cross-fit provenance and area calculation

**Files:**
- Create: `kember_metrics/src/hcp7t_crossfit_area.py`
- Create: `kember_metrics/work/.gitignore` (generated HCP cache is not versioned)
- Modify: `kember_metrics/config/analysis_config.json`
- Test: `kember_metrics/tests/test_hcp7t_crossfit_area.py`

**Interfaces:**
- `load_hcp7t_area_features(config, output_dir, workbench_command) -> tuple[pd.DataFrame, dict[str, object]]`
- `compute_hcp7t_reference_effect(hcp_features, nthc_features) -> pd.DataFrame`
- The returned feature rows contain `subject`, `cohort="HCP 7T"`, `method="hcp151_crossfit_template_matching"`, `smoothing_mm=2`, `hemisphere`, `system`, and `surface_area_mm2`.
- The returned provenance records the cross-fit manifest, five-fold/family-aware contract, held-out count, native area convention, and exact source/label paths.

- [x] **Step 1: Write failing tests** for rejecting a non-family-aware or non-five-fold HCP manifest, accepting a held-out combo manifest only when `subject_was_in_template` is false, and producing the HCP 7T minus NTHC effect from two small real DataFrames.
- [x] **Step 2: Run the focused test file** with `/home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/test_hcp7t_crossfit_area.py -q`; observe failure because the new module/interface is absent.
- [x] **Step 3: Implement the minimum loader** that validates `crossfit_manifest.json` (`n_subjects=151`, `n_folds=5`, family-aware true), validates each combo manifest's held-out status, loads cross-fit labels and HCP native `surfareacorobl`, calls `align_labels_to_native`, and calls `compute_system_surface_area`.
- [x] **Step 4: Implement the explicit Welch reference effect** as HCP 7T minus Stanford NTHC for each hemisphere/system, including mean difference, Welch CI, Cohen's d, and raw p-value without altering Stanford result files.
- [x] **Step 5: Run the focused tests** again and require all new tests to pass.

### Task 2: Add the HCP reference to the PNG figure and remove layout collisions

**Files:**
- Modify: `kember_metrics/src/plot_area_2mm_comprehensive_summary.py`
- Test: `kember_metrics/tests/test_area_summary.py`

**Interfaces:**
- Preserve the seven configured Stanford cohort constants and add `HCP 7T` only to a separate display order.
- Add `HCP 7T−NTHC` only to the rendered effect order.
- Keep `render_area_summary(...)` backward-compatible for the existing Stanford-only test fixture while accepting optional HCP reference inputs.
- Return manifest fields for the display order, HCP cross-fit method, HCP subject count, HCP comparison, and explicit spacing values.

- [x] **Step 1: Extend tests first** to require the HCP display group/contrast, to verify optional HCP rows are included in raw distributions, to verify the manifest records cross-fit provenance, and to verify the layout contract has compact vertical spacing, a reserved left margin, and horizontal wrapped x labels.
- [x] **Step 2: Run the focused area-summary tests** and observe the expected failure for the missing HCP display/layout contract.
- [x] **Step 3: Implement the minimum render changes**: use a wider/taller figure, explicit `GridSpec` margins/`hspace`/`wspace`, wrapped horizontal group labels, extra room above raw-distribution significance brackets, a separate HCP color/label, and light-gray horizontal separators between groups with no footer or unexplained forest x-grid. Keep one `savefig` call targeting `.png` only.
- [x] **Step 4: Wire the CLI** to load HCP features/effects from the configured cross-fit branch and combine them only for plotting.
- [x] **Step 5: Run the focused area-summary tests** until green and run the CLI once against the maintained local HCP outputs.
- [x] **Step 6: Inspect the final PNG with the image viewer** at its exported resolution; confirm titles, forest labels, raw x labels, brackets, legend, group separators, and HCP rows do not collide or clip; the footer is intentionally absent.

### Task 3: Synchronize documentation and validate the delivered artifact

**Files:**
- Modify: `kember_metrics/README.md`
- Modify: `docs/restructure_record.md`
- Modify: `kember_metrics/figures/retained_exploratory/README.md`
- Modify: `kember_metrics/figures/retained_exploratory/figure_provenance.json`

- [x] **Step 1: Document** that the current summary uses seven Stanford cohorts plus a separate HCP 7T reference, that HCP uses family-aware five-fold held-out template matching, and that the HCP comparison is descriptive rather than part of the Stanford FDR family.
- [x] **Step 2: Document** the PNG-only output and the new config/source paths; PDF figure artifacts are not retained or generated.
- [x] **Step 3: Run the complete Kember test suite** with `/home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests -q`.
- [x] **Step 4: Run `py_compile`, `git diff --check`, and a post-render check for the PNG and absence of the comprehensive PDF sibling.
- [x] **Step 5: Review the staged path list, commit the project-only change, and verify the project subdirectory is clean after the commit.

## Self-Review Checklist

- HCP labels are read from per-subject cross-fit outputs, not from `outputs_hcp/templates/hcp151_routea_full_k3_fisherz/`.
- Every HCP subject is checked against its combo manifest's held-out contract.
- Native HCP area uses the same `surfareacorobl` and native-label transfer convention as the Stanford Kember endpoint.
- The HCP comparison is clearly separated from Stanford's six clinical contrasts and FDR family.
- The final artifact is a readable PNG with no generated PDF.

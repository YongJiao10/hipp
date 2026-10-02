# Iowa3T Two-Group Core-Workflow Overviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Recompute the Iowa3T subgroup core workflow for the 7-subject pure-MDD group and the 19-subject comorbid-MDD group, producing four validated overview PNGs per group.

**Architecture:** Reuse the validated subject-level HippUnfold surfaces, 22P RouteA references, and 22P surface time series already materialized for the 94-subject JJ_TMS cohort. Write isolated subgroup FC, gradient, group-atlas, weighted-Fisher-z, and presentation roots so the old 27-subject `routea_patient` outputs are untouched.

**Tech Stack:** Existing numbered `scripts/core_workflow` Python entrypoints, JJ_TMS shared inputs, Kong2022 RouteA FC, HippoMaps/BrainSpace gradients, group-atlas K=2..8 diagnostics with primary K=3, and `py314`.

**Spec:** `docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md` and the user-approved Iowa3T grouping: pure MDD `n=7`, comorbid MDD `n=19`, with `JJ_TMS_3003` excluded.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for every repository Python command.
- Active system is IDAS; keep total workflow concurrency at or below 10. Use `--jobs 1` for K evaluation and at most `--jobs 2` for FC materialization.
- The diagnostic/processing source is `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx`, worksheet `Sheet1`; the 94-subject roster is `QC == "Yes"`.
- Use new isolated names `pure_mdd` and `comorbid_mdd`; never write subgroup FC into the existing `shared_inputs/routea` or `shared_inputs_22p/routea`, and never write subgroup outputs into `routea_patient`.
- The naming convention is fixed: no suffix means the default 36P branch; the 22P subgroup branch must append `_22p` to FC, gradient, group-atlas, weighted, and presentation roots.
- The four overview PNGs are the three base products (gradient-spectral Fisher-z, gradient-spectral normalized Top-20 count, WTA normalized Top-20 count) plus the weighted-Fisher-z gradient-spectral comparison.
- Do not delete or overwrite existing output roots. The new roots must be absent or independently resumable before execution.


## Execution record

The completed rerun uses the already validated 22P data branch:

- Read-only reference/surface/mask inputs: `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs_22p/{routea,surface}/`.
- New 22P subgroup FC roots: `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/routea_{pure_mdd,comorbid_mdd}_22p/`.
- The unversioned `shared_inputs/routea` root is the 36P comparison input and must not be described as 22P.
- Completed 22P products are the two `routea_{pure_mdd,comorbid_mdd}_22p` gradient/group-atlas/weighted roots and their presentation directories; the corresponding no-suffix roots are reserved for 36P.

---

### Task 1: Pin subgroup manifests and documentation

**Files:**
- Create: `manifests/jj_tms_pure_mdd_7_subjects.txt`
- Create: `manifests/jj_tms_comorbid_mdd_19_subjects.txt`
- Modify: `docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md`
- Modify: `scripts/jj_tms/README.md`

**Interfaces:**
- Consumes: The 27-patient roster and `subject_infomation.xlsx` labels.
- Produces: Two ordered bare-ID manifests accepted by `04_01`, `05_01`, and `05_02`; docs use the exact label `comorbid MDD`.

- [x] **Step 1: Write the two ordered manifests**

Pure MDD IDs:
`3010, 3011, 3015, 3017, 3018, 3020, 3023`.

Comorbid MDD IDs:
`3001, 3002, 3004, 3005, 3006, 3007, 3008, 3009, 3012, 3013, 3014, 3016, 3019, 3022, 3024, 3025, 3026, 3027, 3028`.

- [x] **Step 2: Validate roster disjointness and source labels**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c 'from pathlib import Path; from openpyxl import load_workbook; p=Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/subject_infomation.xlsx"); w=load_workbook(p, read_only=True, data_only=True)["Sheet1"]; r={x[0]:x for x in w.iter_rows(min_row=3, values_only=True) if x[0]}; a={x.strip() for x in Path("manifests/jj_tms_pure_mdd_7_subjects.txt").read_text().splitlines() if x.strip() and not x.startswith("#")}; b={x.strip() for x in Path("manifests/jj_tms_comorbid_mdd_19_subjects.txt").read_text().splitlines() if x.strip() and not x.startswith("#")}; assert len(a)==7 and len(b)==19 and not a&b and "3003" not in a|b and "3021" not in a|b; assert all(r["JJ_TMS_"+x][10]=="Yes" for x in a|b); print("manifest validation passed", len(a), len(b))'
```

- [x] **Step 3: Update the cohort documentation**

Use `comorbid MDD` as the short group label and retain the exact operational definition `Current MDD with GAD and/or PTSD comorbidity` in the source/provenance note.

---

### Task 2: Materialize isolated subgroup FC caches

**Files:**
- Create: `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/routea_pure_mdd_22p/`
- Create: `outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/routea_comorbid_mdd_22p/`

**Interfaces:**
- Consumes: Existing 22P `shared_inputs_22p/routea` references, 22P `shared_inputs_22p/surface` masks/timeseries, and the two manifests.
- Produces: Four `kong2022` FC combinations per subject in each isolated root.

- [x] **Step 1: Run FC materialization for pure MDD**

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py   --shared-root outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs/routea_pure_mdd_22p   --surface-root outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs_22p/surface   --reference-root outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs_22p/routea   --mask-root outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/shared_inputs_22p/surface   --subjects-manifest manifests/jj_tms_pure_mdd_7_subjects.txt   --atlas kong2022 --smoothings 2mm 4mm --hemis L R --jobs 2
```

- [x] **Step 2: Run FC materialization for comorbid MDD**

Use the same command with `routea_comorbid_mdd_22p` and `jj_tms_comorbid_mdd_19_subjects.txt`.

- [x] **Step 3: Verify each subgroup has 7/19 subjects and 4 FC combinations per subject**

Check every subject has `fc/kong2022/{2mm,4mm}/hemi_{L,R}/hipp_vertex_to_network_fc.npy` and `fc_summary.json`; fail if any expected subject or combination is missing.

---

### Task 3: Compute subgroup HippoMaps gradients

**Files:**
- Create: `outputs_iowa_jj_tms/work/hippomaps_gradients/routea_pure_mdd_22p/`
- Create: `outputs_iowa_jj_tms/work/hippomaps_gradients/routea_comorbid_mdd_22p/`
- Create: `manifests/core_workflow/05_01_iowa3t_pure_mdd_22p.json`
- Create: `manifests/core_workflow/05_01_iowa3t_comorbid_mdd_22p.json`

**Interfaces:**
- Consumes: Isolated subgroup FC roots from Task 2, shared surface timeseries, and the existing 151-subject group-average HippUnfold surfaces.
- Produces: Aligned G1-G5 arrays and gradient manifests for 2mm/4mm × L/R.

- [x] **Step 1: Run `05_01` for pure MDD**

Forward `--routes routea --smoothings 2mm 4mm --hemis L R --n-components 5 --gradient-random-state 0 --selected-components 1,2,3`, with subgroup-specific `--shared-inputs-root`, `--subjects-manifest`, and `--out-root`; set `--group-average-surfaces-dir outputs_iowa_jj_tms/inputs/iowa_jj_tms/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces`.

- [x] **Step 2: Run `05_01` for comorbid MDD**

Repeat with the comorbid roots and manifest.

- [x] **Step 3: Verify all eight gradient manifests**

Require 2mm/4mm × L/R output manifests, `n_subjects=7` or `19`, selected components `[1,2,3]`, and no source manifest path pointing to `routea_patient`.

---

### Task 4: Run subgroup group-atlas parcellation

**Files:**
- Create: `outputs_iowa_jj_tms/work/group_atlas/routea_pure_mdd_22p_full_k3_fisherz/`
- Create: `outputs_iowa_jj_tms/work/group_atlas/routea_comorbid_mdd_22p_full_k3_fisherz/`

**Interfaces:**
- Consumes: Isolated subgroup FC roots, subgroup gradients, 22P reference/surface roots, subgroup manifests, and the existing group-average surfaces.
- Produces: Gradient-spectral and WTA group-atlas outputs for 2mm/4mm × L/R, K=2..8 diagnostics, primary K=3, Fisher-z and normalized Top-20 assignments.

- [x] **Step 1: Run `05_02` for pure MDD**

Use `--dataset outputs_iowa_jj_tms --route-tag routea --methods gradient-spectral wta-top --atlases kong2022 --smoothings 2mm 4mm --hemis L R --coverage-thresholds 0.80 --k-min 2 --k-max 8 --default-k 3 --gradient-components 1,2,3 --jobs 1`, with all roots and the pure manifest.

- [x] **Step 2: Run `05_02` for comorbid MDD**

Repeat with the comorbid roots and manifest.

- [x] **Step 3: Validate group-atlas manifests**

Require all eight gradient-spectral and eight WTA smoothing/hemisphere combinations to finish with `status=ok`, the correct `n_subjects`, and the subgroup manifest path.

---

### Task 5: Build weighted-Fisher-z comparison branches

**Files:**
- Create: `outputs_iowa_jj_tms/work/group_atlas/routea_pure_mdd_22p_weighted_fisherz_full_k3/`
- Create: `outputs_iowa_jj_tms/work/group_atlas/routea_comorbid_mdd_22p_weighted_fisherz_full_k3/`
- Create: `outputs_iowa_jj_tms/present/overviews/kong2022_8net/22p/pure_mdd/weighted_fisherz_difference_summary.*`
- Create: `outputs_iowa_jj_tms/present/overviews/kong2022_8net/22p/comorbid_mdd/weighted_fisherz_difference_summary.*`

**Interfaces:**
- Consumes: Each subgroup base group-atlas root, 22P surface timeseries, 22P RouteA references, and the Iowa3T group-average surface/spec.
- Produces: Isolated weighted-Fisher-z labels and comparison manifests without modifying base Fisher-z outputs.

- [x] **Step 1: Run `05_05` for pure MDD**

Pass the pure base root, pure weighted output root, shared surface/reference roots, the pure overview directory, Iowa3T group-average surface directory, and the subject spec for `sub-3001`; derive the weighted Fisher-z `gradient-spectral` outputs. WTA remains the separate `normalized_top20_count` assignment and is not part of the weighted branch.

- [x] **Step 2: Run `05_05` for comorbid MDD**

Repeat with the comorbid roots.

- [x] **Step 3: Verify weighted branch manifests**

Require `status=ok`, two methods, 2mm/4mm × L/R weighted runs, and the correct subgroup subject count.

---

### Task 6: Render four overview PNGs per subgroup

**Files:**
- Create: `outputs_iowa_jj_tms/present/overviews/kong2022_8net/22p/pure_mdd/`
- Create: `outputs_iowa_jj_tms/present/overviews/kong2022_8net/22p/comorbid_mdd/`

**Interfaces:**
- Consumes: Base and weighted group-atlas roots from Tasks 4-5.
- Produces: Three base overview PNGs plus one weighted gradient-spectral PNG per subgroup.

- [x] **Step 1: Render the three base overviews for pure MDD**

Run `06_01_build_group_overview.py` with `--methods gradient-spectral wta-top --assignment-variant both --group-name "pure MDD"`; this must produce Fisher-z gradient, normalized Top-20 gradient, and normalized Top-20 WTA PNGs.

- [x] **Step 2: Render the three base overviews for comorbid MDD**

Repeat with `--group-name "comorbid MDD"`.

- [x] **Step 3: Render the weighted gradient overview for each subgroup**

Run `06_01_build_group_overview.py` with `--methods gradient-spectral --assignment-variant weighted_fisherz --manifest-name group_overview_weighted_fisherz_manifest.json --skip-readme`; use the matching weighted root and group name.

- [x] **Step 4: Verify exactly four PNGs per subgroup**

Verify the base manifest has `count=3`, the weighted manifest has `count=1`, all four PNGs are nonempty, and every rendered title carries `pure MDD (N=7)` or `comorbid MDD (N=19)`.

---

### Task 7: Final provenance and output audit

**Files:**
- Modify: `docs/hipp_parcellation_network/jj_tms_hcp151_fixed_template_matching.md` if generated paths or final filenames need recording.
- Modify: `scripts/jj_tms/README.md` if the short label needs a cross-reference.

**Interfaces:**
- Consumes: All manifests and overview products from Tasks 1-6.
- Produces: A reproducible audit report in the final handoff; no unrelated files are changed.

- [x] **Step 1: Check new roots and titles against the two manifests**

Run a py314 JSON/PNG audit that checks subject counts, exact IDs, source roots, manifest counts, PNG sizes, and title labels.

- [x] **Step 2: Check scoped diff and status**

Run `git diff --check` on changed repository files and report unrelated pre-existing worktree changes separately.

- [x] **Step 3: Report output paths**

Return the two overview directories and enumerate each of the four PNGs, including the 7/19 subject count and the excluded `JJ_TMS_3003` note.

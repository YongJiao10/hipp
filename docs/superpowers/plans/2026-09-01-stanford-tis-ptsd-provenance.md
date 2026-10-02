# Stanford TIS PTSD Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the Stanford `tis_ptsd` subgroup reproducibly from the Master Database `Current_PTSD_Diagnosis` field intersected with the existing QC-passed TIS roster, preserve machine-readable provenance, and wire the verified cohort into the Stanford fast-rest direct-comparison workflow.

**Architecture:** Add a small standard-library-only clinical-roster module with a pure derivation function and a CLI that writes the subject manifest plus deterministic provenance JSON. The direct workflow will consume that generated manifest as a nested seventh cohort while the formal four-cell factorial workflow remains unchanged; tests will verify both the derivation rule and the live 35-subject result.

**Tech Stack:** Python 3.14 at `/home/yojiao/.conda/envs/py314/bin/python`, `csv`, `json`, `hashlib`, `argparse`, pytest, existing JSON/TSV manifest conventions.

**Spec:** Current-session user request: derive Stanford `tis_ptsd` from `data/CC_RC_Master_Database_10.23.18_Data.csv` using `CC_Post_Intake_ID` and `Current_PTSD_Diagnosis`, preserve the missing provenance, and verify the subgroup size.

## Global Constraints

- Do not inspect archive content; use only the repository Master Database and non-archive inputs.
- Use the existing QC-passed TIS manifest as the analysis-eligibility universe; do not infer eligibility from diagnosis rows alone.
- Treat `Current_PTSD_Diagnosis == "1"` as positive and fail loudly on duplicate/missing TIS IDs or unexpected diagnosis codes.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for every repository Python task; do not use bare `python` for the task.
- Preserve unrelated dirty-worktree changes and do not regenerate large analysis outputs in this change.
- Update documentation whenever behavior, inputs, manifests, or output paths change.
- Do not add `tis_ptsd` to the formal four-cell `NTHC/TEHC/NTS/TIS` factorial roster; it is a nested diagnostic subgroup for the direct cohort workflow.

---

### Task 1: Define the clinical derivation contract with failing tests

**Files:**
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_clinical_rosters.py`

**Interfaces:**
- The tests define `clinical_rosters.derive_tis_ptsd_roster(master_csv, tis_manifest)` as returning a result with `subject_ids`, `source_rows`, `tis_n`, and provenance fields sufficient to write the manifest and evidence JSON.

- [ ] **Step 1: Write the failing tests**

  Add tests for:

  ```python
  def test_derives_current_ptsd_only_inside_tis(tmp_path):
      # Build a tiny CSV containing TIS and non-TIS rows, including current PTSD 0/1.
      # Assert only TIS rows with Current_PTSD_Diagnosis == "1" are selected.
  
  def test_rejects_duplicate_or_missing_tis_post_ids(tmp_path):
      # Assert malformed TIS-to-database joins raise ValueError instead of silently
      # changing the cohort size.
  
  def test_repository_tis_ptsd_derivation_has_35_members():
      result = derive_tis_ptsd_roster(REPO_ROOT / "data/CC_RC_Master_Database_10.23.18_Data.csv", TIS_MANIFEST)
      assert result.tis_n == 47
      assert len(result.subject_ids) == 35
      assert result.source_id_field == "CC_Post_Intake_ID"
      assert result.diagnosis_field == "Current_PTSD_Diagnosis"
  ```

- [ ] **Step 2: Run the focused tests to verify the RED state**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_clinical_rosters.py -q
  ```

  Expected result: collection fails because `clinical_rosters` does not yet exist. Confirm the failure is about the missing implementation, not a test typo.

### Task 2: Implement reproducible CSV-to-roster derivation

**Files:**
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/src/clinical_rosters.py`
- Test: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_clinical_rosters.py`

**Interfaces:**
- `derive_tis_ptsd_roster(master_csv: Path, tis_manifest: Path, positive_code: str = "1") -> ClinicalRosterResult`
- `write_roster_outputs(result: ClinicalRosterResult, manifest_path: Path, provenance_path: Path) -> None`
- CLI: `python clinical_rosters.py --master-csv ... --tis-manifest ... --manifest-out ... --provenance-out ...`

- [ ] **Step 1: Implement only the minimum behavior needed by the tests**

  Read the CSV with `csv.DictReader`, normalize numeric IDs to four-digit subject IDs, index `CC_Post_Intake_ID`, read the existing TIS manifest, require exactly one source row per TIS subject, select rows whose `Current_PTSD_Diagnosis` equals `"1"`, and return sorted IDs plus counts and source metadata.

- [ ] **Step 2: Run the focused tests to verify GREEN**

  Run the same focused pytest command and confirm the synthetic and repository derivation tests pass.

- [ ] **Step 3: Add deterministic evidence writing**

  Write the generated numeric manifest and provenance JSON containing:

  - source path relative to repository root;
  - source SHA-256 and CSV row count;
  - ID field and diagnosis field;
  - positive diagnosis code;
  - TIS manifest path, SHA-256, and `tis_n`;
  - selected `tis_ptsd_n`, selected subject IDs, and excluded TIS IDs;
  - derivation statement and explicit note that the subgroup is nested within TIS.

- [ ] **Step 4: Re-run focused tests after evidence writing**

  Confirm all focused tests remain green and that output serialization is deterministic for identical inputs.

### Task 3: Materialize the canonical roster and provenance evidence

**Files:**
- Create: `outputs_stanford/inputs/stanford/kong2022/manifests/stanford_tis_ptsd_4xxx_current_ptsd_qc_pass_t1only_35.txt`
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/reports/stanford_tis_ptsd_provenance.json`

**Interfaces:**
- The files are generated by Task 2 from the checked-in Master Database and existing TIS manifest; they are not hand-curated subject lists.

- [ ] **Step 1: Run the derivation CLI against the canonical inputs**

  Run from the repository root with explicit paths:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/src/clinical_rosters.py \
      --master-csv data/CC_RC_Master_Database_10.23.18_Data.csv \
      --tis-manifest outputs_stanford/inputs/stanford/kong2022/manifests/stanford_tis_4xxx_qc_pass_t1only_47.txt \
      --manifest-out outputs_stanford/inputs/stanford/kong2022/manifests/stanford_tis_ptsd_4xxx_current_ptsd_qc_pass_t1only_35.txt \
      --provenance-out stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/reports/stanford_tis_ptsd_provenance.json
  ```

- [ ] **Step 2: Verify the generated count and join evidence**

  Assert the manifest has 35 numeric IDs, provenance reports TIS 47 and selected 35, no missing/duplicate source joins, and source diagnosis counts are 35 positive versus 12 negative within TIS.

- [ ] **Step 3: Run the canonical derivation test**

  Run `test_repository_tis_ptsd_derivation_has_35_members` and retain its output as the count verification.

### Task 4: Wire `tis_ptsd` into the direct cohort contract

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/config/direct_contrast_config.json`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/config/direct_contrast_fixed_config.json`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/src/direct_contrasts.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_direct_contrasts.py`

**Interfaces:**
- Extend the direct cohort order to append `TIS PTSD`, configure the 35-member generated manifest, and add `TIS PTSD−NTHC`.
- Preserve existing overlap behavior: `TIS PTSD` is a proper subset of `TIS`, and direct cohort membership is allowed to overlap.

- [ ] **Step 1: Update direct-contrast tests first**

  Change expected cohort order/counts, assert `set(rosters["TIS PTSD"]) < set(rosters["TIS"])`, add `TIS PTSD−NTHC` to contrasts, and update endpoint count from 180 to 216.

- [ ] **Step 2: Run the focused direct-contrast tests to verify RED**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_direct_contrasts.py -q
  ```

  Expected result: failures identify the missing seventh-cohort contract.

- [ ] **Step 3: Update constants/configuration minimally**

  Append the cohort and contrast in both configs and update `COHORT_ORDER`/related documentation strings in `direct_contrasts.py` without changing the statistical estimator.

- [ ] **Step 4: Re-run direct-contrast tests**

  Confirm the updated roster and endpoint contract pass, including the 35-member nested cohort.

### Task 5: Update the standalone area figure contract and documentation

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/src/plot_area_2mm_summary.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `docs/hipp_parcellation_network/stanford_trauma_nontrauma_group_atlas.md`

**Interfaces:**
- The standalone figure must read the same generated manifest and display seven cohorts/ six case contrasts, including `TIS PTSD (n=35)` and `TIS PTSD−NTHC`.
- Documentation must distinguish source-derived diagnostic rosters from the four primary factorial cells and link the provenance JSON.

- [ ] **Step 1: Update hardcoded visual labels/configuration**

  Add group color, roster entry, contrast entry, bracket position, and count text for `TIS PTSD`; update six/seven and five/six labels while preserving plot geometry and non-destructive output behavior.

- [ ] **Step 2: Document the derivation contract**

  State that `tis_ptsd` is the intersection of the existing 47-person TIS QC roster with `Current_PTSD_Diagnosis == 1` in `data/CC_RC_Master_Database_10.23.18_Data.csv`, keyed by `CC_Post_Intake_ID`; link the generated manifest and provenance report; state that the existing Pure MDD roster remains legacy fixed input unless separately re-derived.

- [ ] **Step 3: Run documentation/configuration checks**

  Parse both JSON configs, search for stale “five contrasts/six cohorts” wording in the touched workflow docs/source, and verify every referenced manifest/provenance path exists.

### Task 6: Full verification and handoff

**Files:**
- Verify only; no additional source files unless a test exposes a defect.

- [ ] **Step 1: Run the focused clinical and direct tests**

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest \
      stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_clinical_rosters.py \
      stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_direct_contrasts.py -q
  ```

- [ ] **Step 2: Run the broader fast-rest tests relevant to roster/config contracts**

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests -q
  ```

- [ ] **Step 3: Recompute the live evidence independently**

  Re-run the derivation CLI or a read-only validation command and record: source rows 332, TIS rows 47, current PTSD positives 35, current PTSD negatives 12, missing joins 0, duplicate joins 0.

- [ ] **Step 4: Inspect the final diff and status**

  Use `git diff --` on only the touched files and `git status --short` to confirm unrelated existing deletions/modifications were preserved. Do not claim completion unless all required verification commands exit 0.

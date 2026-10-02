# Fixed HCP Identity Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three group-dependent/low-interpretability identity metrics with two fixed-HCP-territory metrics and regenerate the Standard and Ctx-Reg identity figures.

**Architecture:** The HCP151 template manifest supplies the exact, branch- and hemisphere-specific A/B/P labels and support mask. Each subject's already-existing three HCP-template scores are area-weighted only within those fixed HCP territories. The old group-dominant identity extraction is removed; group atlases remain only where they are required for descriptive maps and fingerprint analyses.

**Tech Stack:** Python 3.14, NumPy, unittest, Matplotlib.

**Spec:** `scripts/visualize/iowa_hipp_functional_organization/README.md`

## Global Constraints

- Use the exact HCP151 template-manifest paths for `standard → routea` and `ctxreg → routea_ctxreg`; no artifact discovery or fallback.
- Run local Python work with `/home/yojiao/.conda/envs/py314/bin/python` and one numerical-library thread.
- The identity panel must contain exactly six metrics: `smax`, `margin`, `fixed_hcp_similarity`, `fixed_hcp_advantage`, `segregation`, and `boundary_sharpness`.
- Remove `all_negative_fraction`, `designated_similarity`, and group-dominant `designated_advantage` from newly computed outputs, identity figures, and documentation.
- Render only `hipp_functional_identity_standard.png` and `hipp_functional_identity_ctxreg.png` after recomputing the derived metric tables.

---

### Task 1: Fixed HCP territory metric and input contract

**Files:**

- Modify: `scripts/visualize/iowa_hipp_functional_organization/metrics.py`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/inputs.py`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/analysis.py`
- Test: `scripts/visualize/iowa_hipp_functional_organization/test_metrics.py`
- Test: `scripts/visualize/iowa_hipp_functional_organization/test_analysis.py`

**Interfaces:**

- Produces `metrics.fixed_hcp_identity(scores, hcp_labels, hcp_support, area, valid=None)`.
- Produces two scalar rows per A/B/P HCP territory: `fixed_hcp_similarity` and `fixed_hcp_advantage`.
- Adds exact HCP labels/support/prototype records to the validated inventory and passes them to each subject worker.

- [ ] **Step 1: Write failing tests**

```python
rows = self.m.fixed_hcp_identity(
    scores, hcp_labels, hcp_support, area, valid=valid
)
self.assertAlmostEqual(self.pick(rows, "A", "fixed_hcp_similarity")["value"], 0.5)
self.assertAlmostEqual(self.pick(rows, "A", "fixed_hcp_advantage")["value"], 0.3)
```

Add a malformed HCP support/label assertion and an analysis-contract assertion that a fixed HCP index rejects duplicate or malformed branch/hemi records.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python -m unittest scripts.visualize.iowa_hipp_functional_organization.test_metrics scripts.visualize.iowa_hipp_functional_organization.test_analysis
```

Expected: failure because `fixed_hcp_identity` and the HCP-template inventory/index contract do not exist.

- [ ] **Step 3: Implement the minimal contract**

Read only the exact `template_manifest.json` selected by the branch. Require HCP151, density 512, 400 ROI features, complete status, fixed A/B/P semantics, one 2 mm record for each hemisphere, binary support, and `labels != 0` exactly equal to support. Use HCP territory area as `total_area` and evaluable subject area as `valid_area`.

- [ ] **Step 4: Run tests to verify they pass**

Run the Step 2 command and require zero failures.

### Task 2: Six-panel identity renderer and contract documentation

**Files:**

- Modify: `scripts/visualize/iowa_hipp_functional_organization/report.py`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/test_report.py`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/README.md`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/numerical_notes.md`
- Modify: `scripts/visualize/iowa_hipp_functional_organization/report_notes.md`

**Interfaces:**

- Identity renderer reads the six declared metrics only.
- `analysis_config.json` declares the fixed-HCP identity metric definition.
- Renderer rejects a derived table that lacks any required identity panel metric or retains an obsolete identity metric.

- [ ] **Step 1: Write a failing report-contract test**

Create a conforming synthetic identity dataset containing the six required metrics, then remove `fixed_hcp_advantage` and assert that report loading fails with an identity-metric coverage error.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python -m unittest scripts.visualize.iowa_hipp_functional_organization.test_report
```

Expected: failure because the current renderer neither requires fixed-HCP metrics nor rejects the retired metrics.

- [ ] **Step 3: Implement the six-panel renderer**

Replace the three retired metrics with fixed-HCP similarity and advantage in the figure grid, captions, y-axis handling, `METRIC_INFO`, and guide card. Keep the four-by-two layout: six metric panels and a guide in the eighth cell. Record definitions and HCP manifest provenance in the README/numerical/report notes.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command and require zero failures.

### Task 3: Recompute and render the two requested figures

**Files:**

- Modify: `scripts/visualize/iowa_hipp_functional_organization/analysis.py`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/subject_metrics.csv`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/statistics.csv`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/analysis_config.json`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/input_manifest.json`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/input_audit.csv`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/run_manifest.json`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/hipp_functional_identity_standard.png`
- Update generated: `scripts/visualize/iowa_hipp_functional_organization/results/hipp_functional_identity_ctxreg.png`

**Interfaces:**

- Add an identity-only render stage that reads the current validated derivatives and records hashes for only the two identity PNGs.

- [ ] **Step 1: Add a failing parser/identity-render test**

Assert that `--stage identity` is accepted and uses only the two identity PNG output names.

- [ ] **Step 2: Run the affected unit tests to verify failure**

Run the Task 1 and Task 2 test commands. Expected: parser rejects the new stage or no identity-only renderer exists.

- [ ] **Step 3: Implement and run derived stages**

Run inventory and analysis with one worker and explicit overwrite, then run the identity-only render stage with explicit overwrite. Do not render unrelated PNGs, HTML, or PDF.

- [ ] **Step 4: Verify outputs**

Run all scoped tests, assert retired metric identifiers are absent from the regenerated identity CSV/statistics, assert the two PNGs exist and have nonzero size, and inspect each generated image.


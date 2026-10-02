# Stanford Individual Rest-FC Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Stanford-only, individual-level Fisher-z FC evidence summary that separates inferential findings from non-inferential group-atlas descriptions.

**Architecture:** Read only the sealed attempt8 subject-level localisation and four-group descriptive tables. Aggregate fixed HCP151 and individual template-matching specifications as correlated sensitivity views, write non-overwriting derivative tables and a Chinese scientific report, then link the result from active README files. The group-atlas branch is named only as descriptive context and is excluded from every inferential result table.

**Tech Stack:** Python 3.14, pandas, sealed CSV.GZ tables, Markdown.

**Spec:** `stanford_jjtms_multiscale_hippocampal_exploration/rest_fc_primary_analysis_spec.md`

## Global Constraints

- Resting-state Fisher-z FC only; no beta, TMS-response, CON, native stimulation, or archive inputs.
- Use only `hcp151_fixed` and `hcp151_template_matching`.
- Treat method, smoothing, and hemisphere specifications as correlated sensitivity views, not independent replications.
- Keep `primary_screen_v1_20260824_attempt8` sealed and read-only.
- Stanford group-atlas outputs have no individual-level inference and must not appear as individual FC discoveries.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python work.
- Do not delete or overwrite historical outputs.

---

### Task 1: Audit and freeze the individual-level evidence scope

**Files:**
- Read: `cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt8/primary_localisation.csv.gz`
- Read: `cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt8/stanford_four_group_descriptives.csv.gz`
- Create: `cross_dataset_rest_fc/reports/stanford_individual_results_v1_20260827/source_scope.md`

**Consumes:** Sealed Stanford subject-level Fisher-z FC rows and four group descriptive rows.

**Produces:** A source scope record with the exact input paths, n=194 cell counts, retained methods, retained hemisphere modes, and explicit exclusion of group-atlas maps.

- [ ] **Step 1: Verify source schemas and nonmissing subject-level rows**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import pandas as pd; p='stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt8/primary_localisation.csv.gz'; d=pd.read_csv(p); q=d[(d.dataset=='stanford') & d.method.isin(['hcp151_fixed','hcp151_template_matching']) & d.failure_reason.isna()]; assert len(q)==8400; print(q.groupby('estimand').size().to_dict())"
```

Expected: Seven Stanford estimands and 8,400 finite subject-level localisation rows.

- [ ] **Step 2: Record the reporting taxonomy**

Write a short scope record that distinguishes:
- level FC: bilateral, L, and R, 12 correlated specifications per component;
- laterality FC: paired within-subject L-R, 4 correlated specifications per component;
- group-atlas: excluded from individual-level inference.

- [ ] **Step 3: Verify scope record contents**

Run:

```bash
rg -n "individual|group-atlas|hcp151_fixed|hcp151_template_matching|L-R" cross_dataset_rest_fc/reports/stanford_individual_results_v1_20260827/source_scope.md
```

Expected: All five scope constraints are stated.

### Task 2: Build complete Stanford individual-result tables and report

**Files:**
- Create: `cross_dataset_rest_fc/results/stanford_individual_results_v1_20260827/stanford_individual_component_summary.tsv`
- Create: `cross_dataset_rest_fc/results/stanford_individual_results_v1_20260827/stanford_individual_priority_findings.tsv`
- Create: `cross_dataset_rest_fc/results/stanford_individual_results_v1_20260827/stanford_individual_estimand_overview.tsv`
- Create: `cross_dataset_rest_fc/reports/stanford_individual_results_v1_20260827/stanford_individual_results.md`
- Create: `cross_dataset_rest_fc/results/stanford_individual_results_v1_20260827/run_manifest.json`

**Consumes:** The Task 1 sealed rows and four-group descriptive rows.

**Produces:** Complete aggregate tables for all seven Stanford estimands, prioritized individual-level findings, four-group mean patterns for named candidates, and a source/output hash manifest.

- [ ] **Step 1: Calculate level and laterality summaries**

For each `estimand × resolution × region × network`, calculate median standardized effect, min/max effect, sign consistency, min BH q, method-specific medians, and number of q<.05/.10 rows. Use 12 level specifications for bilateral/L/R and 4 specifications for L-R separately.

- [ ] **Step 2: Make prioritization explicit**

Label rows as:
- `FDR-supported`: at least one family-specific q<.05 row;
- `stable effect candidate`: all level specifications agree in sign and absolute median d>=.35;
- `laterality candidate`: all L-R specifications agree in sign and absolute median d>=.35;
- `method-sensitive secondary`: all level signs agree, absolute median d<.35, and at least one q<.10 row.

Do not label any group-atlas comparison as a finding.

- [ ] **Step 3: Report all estimands, not just top rows**

The Markdown report must have one compact section each for symptom marginal, trauma marginal, interaction, NTS-NTHC, TIS-TEHC, TEHC-NTHC, and TIS-NTS. Each section names its best supported/stable candidate and states when there is no FDR-supported component.

- [ ] **Step 4: Verify tables and report**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import pandas as pd; r='stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/stanford_individual_results_v1_20260827'; d=pd.read_csv(r+'/stanford_individual_component_summary.tsv', sep='\t'); assert set(d.estimand)=={'symptom_marginal','trauma_marginal','interaction','NTS-NTHC','TIS-TEHC','TEHC-NTHC','TIS-NTS'}; print(d.shape)"
```

Expected: All seven estimands appear and output rows are finite.

### Task 3: Update active documentation

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/.superpowers/sdd/rest_fc_primary_analysis_implementation_plan/progress.md`

**Consumes:** Task 2 report and tables.

**Produces:** An explicit Stanford individual-level evidence link and an accurate separation between individual FC and group-atlas description.

- [ ] **Step 1: Add the Stanford individual-results report as the authoritative individual summary**

Link the report and table directory from the root study README and state that its findings derive from individual Fisher-z FC values.

- [ ] **Step 2: State group-atlas limitations once, clearly**

Keep the group-atlas section, but state it is descriptive only and neither contributes p/q nor replaces individual-level FC evidence.

- [ ] **Step 3: Remove stale scope language from active documentation**

Ensure current active documents do not list out-of-scope parcellation or intervention branches as part of this Stanford individual FC report.

- [ ] **Step 4: Verify documentation links and scope wording**

Run:

```bash
rg -n "Stanford.*individual|individual-level|group-atlas|stanford_individual_results_v1_20260827" stanford_jjtms_multiscale_hippocampal_exploration/README.md stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/README.md
```

Expected: Individual FC is primary and group atlas is descriptive.

### Task 4: Validate traceability and handoff

**Files:**
- Verify: sealed attempt8 run manifest
- Verify: Task 2 run manifest
- Verify: active README files

**Consumes:** All Task 1-3 outputs.

**Produces:** Fresh evidence that inputs, outputs, documentation, and sealed data are consistent.

- [ ] **Step 1: Recompute source and derivative hashes**

Verify every hash recorded in the derivative manifest and verify all 38 sealed attempt8 outputs still match the sealed manifest.

- [ ] **Step 2: Run focused document checks**

Run `git diff --check` on touched Markdown files and scan only active planning/runbook documents for scope-forbidden terms.

- [ ] **Step 3: Report findings in two layers**

Provide:
- a plain-language list of Stanford individual-level results, each beginning with the branch and group contrast;
- a short priority list separating FDR-supported, stable exploratory, laterality, and method-sensitive findings.

## Self-Review

- Spec coverage: Task 1 defines the individual-only evidence boundary; Task 2 provides all seven Stanford estimands and source tables; Task 3 synchronizes active docs; Task 4 verifies integrity and presents the requested findings.
- Placeholder scan: no TBD/TODO or unspecified interfaces remain.
- Type consistency: all Task 2 outputs use the same run id, source paths, and estimator vocabulary.

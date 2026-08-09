# HIPP Project Report Deck Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checklist syntax for tracking.

**Goal:** Rebuild the English HIPP project report as a dense, portable workflow-and-data guide for PI and lab-member handoff, without exposing personal paths or presenting deprecated work as active.

**Architecture:** Keep one editable Python deck generator and one figure registry under `docs/presentations/`, but ignore that entire directory so generated slides and previews are never Git-tracked. Reorder the deck so individual cortical parcellation routes precede the full `core_workflow/` map; then explain valid branch combinations, active analysis methods, data holdings, progress, and planned work.

**Tech Stack:** Python-pptx generator; project Markdown/runbooks; `markitdown`; office package validator; static slide renderer.

## Global Constraints

- Do not add or stage any slide-related file with Git; `.gitignore` must ignore `docs/presentations/`.
- Use no dates, slide numbers, footer labels, `/home/...`, `/Users/...`, `yojiao`, or user-specific interpreter paths in visible slide text.
- Commands shown to future users use `python core_workflow/...` and generic placeholders only; no absolute Python executable.
- Display `LSS/jianglab/3_Data_Working/...` rather than a local mount prefix.
- The directory map starts with `core_workflow/`, never `scripts/core_workflow/`, and lists every numbered stage `01_01` through `07_01`, including all `06_01` through `06_10` entries.
- `04_02` and `04_03` are shown only as numbered `(deprecated)` legacy entries plus their complete directory-tree names; no slide describes Kong2019, intrinsic-spectral, or network-spectral as active work.
- Lynch NSI is an optional exploratory utility, not a formal 78-participant workflow. The report’s primary HCP cohort is the 151 QC-passed participants.
- Do not mention therapeutic iEEG because no registered dataset is available.
- Keep every scientific figure whole (`contain`, no crop), source-registered, context-matched, and embedded at most once.
- Use white academic styling with safe Office fonts; use the largest text that fits each content block (titles at least 34 pt, ordinary body text at least 14 pt, ordinary commands at least 12 pt).

---

### Task 1: Prevent Slide Products from Being Git-Tracked

**Files:**
- Modify: `.gitignore`
- Verify: repository Git ignore rules

**Interfaces:**
- Consumes: existing untracked `docs/presentations/` directory.
- Produces: an ignore rule that covers deck files, previews, figure registry, and generator source in that directory without deleting any existing file.

- [x] **Step 1: Add the presentation-directory ignore rule**

Append this exact rule after the existing output-root rules:

```gitignore
docs/presentations/
```

- [x] **Step 2: Verify the ignore contract**

Run:

```bash
git check-ignore -v docs/presentations/hipp_project_report.pptx
```

Expected: output names `.gitignore` and `docs/presentations/`.

- [x] **Step 3: Confirm no staging action occurred**

Run:

```bash
git diff --cached --name-only
```

Expected: no presentation file is listed.

### Task 2: Define the New Information Architecture from Verified Project Facts

**Files:**
- Modify: `docs/presentations/build_hipp_project_report.py`
- Modify: `docs/presentations/hipp_project_report_figure_registry.md`

**Interfaces:**
- Consumes: `scripts/core_workflow/README.md`, `step_registry.json`, the ArealMSHBM launcher README, and verified data-root counts.
- Produces: a 28–30 slide English report whose sections can be used independently as a project guide.

- [x] **Step 1: Insert two pre-workflow cortical-parcellation slides**

Use this route structure:

```text
Kong2022 ArealMSHBM
  GitHub: ThomasYeoLab/Kong2022_ArealMSHBM + ThomasYeoLab/CBIG
  Python input preparation → shell/Scheduler launcher → MATLAB engine
  outputs: lh_labels.txt + rh_labels.txt → core_workflow/03_02

Lynch2024 FASTANS PFM
  GitHub: mlueckel/FASTANS
  priors + cortex time series → hard-label component maps → core_workflow/03_03
  20-network original taxonomy or canonical 8-network summary
```

State on the Lynch slide that NSI is optional exploratory filtering and that the report’s working HCP cohort is `n=151` after QC.

- [x] **Step 2: Make the complete core-workflow tree one readable slide**

Show all 29 numbered names in two tree columns under `core_workflow/`. Put an expanded stage responsibility panel on the right that explains inputs, QC, individual cortex references, FC/control construction, atlas methods, presentation products, and cross-branch matrix. Mark only `04_02` and `04_03` as `(deprecated)`.

- [x] **Step 3: Explain branches as compatible combinations rather than isolated script names**

Create three decision tables:

```text
Reference/taxonomy: Kong2022 8-network or 17-network; Lynch2024 8-network or 20-network
Control/reference route: individualized reference default; local-cortex regression; local-cortex exclusion; shared cortical reference
Atlas/assignment: gradient-spectral Fisher-z; gradient-spectral count; weighted Fisher-z comparison; WTA normalized top-20 count
```

Mark invalid or unavailable combinations explicitly rather than implying that every axis combines with every other axis.

### Task 3: Replace Personal/Host-Specific Copy with Portable Commands and Paths

**Files:**
- Modify: `docs/presentations/build_hipp_project_report.py`

**Interfaces:**
- Consumes: verified data locations and script entrypoints.
- Produces: commands and paths that a new lab member can adapt without seeing a personal account name.

- [x] **Step 1: Normalize all visible commands**

Use this visible convention:

```text
python core_workflow/05_02_run_group_atlas.py \\
  --methods gradient-spectral wta-top ...
```

The command cards state that paths are relative to the project’s command directory; they do not include a shell prompt, absolute executable, or user directory.

- [x] **Step 2: Normalize visible data references**

Use this exact path style:

```text
LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume/
data/hippunfold_input/
outputs_hcp/inputs/hcp/
hcp_3t_timeseries/
```

- [x] **Step 3: Create the dataset-size table**

Use the audited counts:

```text
HCP 7T: 166 source participants; 151 QC-passed analysis cohort
Stanford: 199 processed participant directories; current group splits 91 HC / 103 symptom
ESTT: 25 processed participant directories; not run
Iowa7T: 3 processed participant directories; clinical data, no results reported
HCP 3T: 1,071 expected/ready subject list; 1,070 found; integration pending
```

For HCP 7T, explicitly show local source assets: T1w/T2w, fsLR32k surfaces, concatenated rest CIFTI in `data/hippunfold_input/`, HippUnfold results under `outputs_hcp/inputs/hcp/hippunfold/`, and 151-person sampled hippocampal time-series/FC caches under `outputs_hcp/inputs/hcp/kong2022/shared_inputs/`.

### Task 4: Rebuild the Visual System for Larger Readable Text

**Files:**
- Modify: `docs/presentations/build_hipp_project_report.py`
- Modify: `docs/presentations/hipp_project_report_figure_registry.md`

**Interfaces:**
- Consumes: content architecture from Tasks 2–3 and six verified source figures.
- Produces: a white-background 16:9 deck with no footer/date, high-density panels, and unique figure use.

- [x] **Step 1: Remove date, footer label, and page-number shapes**

The shared title helper must draw only the section label, title, and optional subtitle. The cover must not include a date.

- [x] **Step 2: Increase typography and reclaim empty space**

Apply these minimum sizes:

```text
slide title: 34–38 pt
section/stage heading: 18–22 pt
body/card text: 14–16 pt
commands and complete tree: 11.5–13 pt
figure caption: 11–12 pt
```

Split a page if content cannot fit at those sizes. Do not reduce ordinary body text to preserve a sparse layout.

- [x] **Step 3: Preserve only source-matched figures**

Use each source image no more than once and only on a page whose method it demonstrates: continuous gradients, gradient-spectral, WTA, local-cortex control, cortex-template strength, and cross-fitted template robustness. Add a source/meaning/limitation panel for every embedded result figure.

### Task 5: Generate and Verify the Revised Deck

**Files:**
- Generate (ignored): `docs/presentations/hipp_project_report.pptx`
- Generate (ignored): `docs/presentations/hipp_project_report_preview_v5/`

**Interfaces:**
- Consumes: revised generator and source figures.
- Produces: a structurally valid deck and static images for visual inspection.

- [x] **Step 1: Generate the deck**

Run:

```bash
python docs/presentations/build_hipp_project_report.py
```

Expected: an ignored `.pptx` and a passing unique-figure audit.

- [x] **Step 2: Run structural and content checks**

Run:

```bash
python /path/to/pptx/scripts/office/validate.py docs/presentations/hipp_project_report.pptx
python -m markitdown docs/presentations/hipp_project_report.pptx | rg -in 'yojiao|/home/|/Users/|technical handbook|iEEG|nsi-filtered|scripts/'
```

Expected: validator success and no search matches.

- [x] **Step 3: Render and inspect every slide**

Run:

```bash
python docs/presentations/render_pptx_static_preview.py docs/presentations/hipp_project_report.pptx docs/presentations/hipp_project_report_preview_v5
```

Inspect the contact sheet and all changed dense slides for clipped text, overflow, unreadably small type, improper figure crop, or unused whitespace. Regenerate after every visual correction.

## Spec Coverage Review

- Git tracking: Task 1 prevents slide products from appearing in status or staging.
- Dates/footer/type size: Task 4 removes the former and enforces the latter.
- Personal paths and portable commands: Task 3 normalizes visible text and Task 5 scans generated content.
- Full `core_workflow/` tree and expanded responsibility: Task 2, Step 2.
- Deprecated Kong2019/intrinsic/network material: Global constraints and Task 2 avoid all active descriptions.
- Combination-based branch explanation: Task 2, Step 3.
- Individual cortical parcellation before core workflow, source resources, and Python/shell/MATLAB chain: Task 2, Step 1.
- Lynch NSI policy: Global constraints and Task 2, Step 1.
- Data sizes and local HCP assets: Task 3, Step 3.
- No unsupported iEEG content: Global constraints.

## Placeholder Scan

No `TODO`, `TBD`, or implicit implementation placeholders are present in this plan.

## Execution Handoff

The user requested the deck revisions directly, so execute this plan inline in the current session rather than pausing for a separate implementation choice.

# Cross-Fit Individual Hippocampal Overviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one compact, standard-Workbench-style PNG per HCP subject that compares their cross-fitted anterior/body/posterior label maps at 2 mm and 4 mm smoothing.

**Architecture:** Add a focused cross-fit presentation entrypoint rather than adapting the unrelated subject-first gradient/WTA adapter. It will validate each held-out subject's existing cross-fit combo manifests and GIFTI labels, render each smoothing with the established locked HippUnfold Workbench scene, then vertically compose the two rendered rows into one subject PNG. The canonical presentation subtree will carry the PNGs plus a machine-readable status TSV and manifest.

**Tech Stack:** Python 3.14, NumPy, nibabel, Pillow, Matplotlib font discovery, Connectome Workbench via the existing `render_locked_grid` helper, pytest.

## Global Constraints

- Canonical output root: `outputs_hcp/present/overviews/crossfit_template_matching/individual/`.
- Filename: `sub-<id>_crossfit_template_matching_k3_overview.png`; each file contains 2 mm and 4 mm rows, with left/right hippocampi rendered by `config/wb_locked_native_view_lateral_medial.scene`.
- Reuse `lib.surface_utils.render_locked_grid`; do not create a competing hippocampal visualization style.
- Validate `status=ok`, subject/smoothing/hemi identity, held-out status, GIFTI label fingerprint, subject surface evidence, label semantics, and label/surface vertex counts before rendering.
- Default worker count is 4 and every execution is capped at 10 concurrent workers.
- Existing PNGs must not be silently overwritten: `--resume` may reuse a validated existing PNG; a non-resume rerun must fail for that subject.
- Document the new command and output tree in the same change as the code.
- Do not commit or delete files unless the user explicitly requests it.

---

## File Structure

- Modify: `scripts/core_workflow/lib/output_layout.py` — expose the canonical individual cross-fit overview resolver.
- Create: `scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py` — validate input labels, invoke the locked renderer, compose 2 mm/4 mm rows, and write status/provenance outputs.
- Create: `tests/individual_workflow/test_crossfit_individual_overviews.py` — test manifest validation, compact two-row composition, non-overwrite/resume behavior, and aggregate status output without invoking Workbench.
- Modify: `scripts/individual_workflow/README.md` — add the supported renderer and its command/output contract.
- Modify: `docs/hipp_parcellation_network/crossfit_template_matching.md` — document the individual visual-output layout and interpretation.

### Task 1: Define the output contract with a failing test

**Files:**
- Create: `tests/individual_workflow/test_crossfit_individual_overviews.py`
- Test: `tests/individual_workflow/test_crossfit_individual_overviews.py`

**Interfaces:**
- Consumes: a synthetic `crossfit_manifest.json`, four synthetic `combo_manifest.json` files, two valid GIFTI labels per smoothing, and synthetic subject surfaces.
- Produces: a failing expectation for `render_subject_overview(...) -> dict[str, object]` and `render_individual_overviews(...) -> dict[str, Path]`.

- [ ] **Step 1: Write the failing test for a two-smoothing subject image**

```python
outputs = overview.render_individual_overviews(
    crossfit_root=crossfit_root,
    output_dir=output_dir,
    scene=scene,
    subjects=["100001"],
    smoothings=["2mm", "4mm"],
    jobs=1,
)

png = output_dir / "sub-100001_crossfit_template_matching_k3_overview.png"
assert png.exists() and png.stat().st_size > 0
assert Image.open(png).height > Image.open(png).width
assert json.loads(outputs["manifest"].read_text())["summary"]["ok"] == 1
```

- [ ] **Step 2: Run the focused test to verify it fails because the module/function is absent**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/individual_workflow/test_crossfit_individual_overviews.py -q
```

Expected: collection/import failure naming `02_01_render_hcp_crossfit_individual_overviews.py` or `render_individual_overviews`.

- [ ] **Step 3: Add failing assertions for provenance and overwrite protection**

```python
item = manifest["items"][0]
assert item["rows"][0]["smoothing"] == "2mm"
assert item["rows"][1]["smoothing"] == "4mm"
assert item["rows"][0]["label_gifti"].endswith(".label.gii")

with pytest.raises(FileExistsError, match="--resume"):
    overview.render_subject_overview(..., resume=False)
```

- [ ] **Step 4: Re-run the focused test and confirm failure is still attributable to missing production behavior**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/individual_workflow/test_crossfit_individual_overviews.py -q
```

Expected: FAIL at the requested renderer API, not at fixture creation.

### Task 2: Implement the canonical renderer and output resolver

**Files:**
- Modify: `scripts/core_workflow/lib/output_layout.py`
- Create: `scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py`
- Test: `tests/individual_workflow/test_crossfit_individual_overviews.py`

**Interfaces:**
- Consumes: `resolve_crossfit_template_matching_root()`, `resolve_crossfit_template_matching_overview_root()`, `render_locked_grid(...)`, cross-fit combo manifests.
- Produces: `resolve_crossfit_template_matching_individual_overview_root() -> Path`, `render_subject_overview(...) -> dict[str, object]`, and `render_individual_overviews(...) -> dict[str, Path]`.

- [ ] **Step 1: Add the output-layout resolver**

```python
def resolve_crossfit_template_matching_individual_overview_root(
    dataset: str = DEFAULT_DATASET,
) -> Path:
    return resolve_crossfit_template_matching_overview_root(dataset) / "individual"
```

- [ ] **Step 2: Implement strict combo-manifest loading before any render**

```python
def load_subject_combo(
    *, crossfit_root: Path, subject: str, smoothing: str, hemi: str,
) -> dict[str, object]:
    manifest_path = crossfit_root / "subjects" / f"sub-{subject}" / smoothing / f"hemi_{hemi}" / "combo_manifest.json"
    manifest = load_json(manifest_path)
    validate_subject_combo_manifest(manifest, subject=subject, smoothing=smoothing, hemi=hemi)
    return {"manifest": manifest_path.resolve(), "label_gifti": label_path.resolve(), "surface": surface_path.resolve()}
```

- [ ] **Step 3: Implement one Workbench render per smoothing and a compact vertical composition**

```python
with tempfile.TemporaryDirectory(prefix="crossfit_individual_overview_") as scratch:
    for smoothing in smoothings:
        row = render_locked_grid(
            subject=subject,
            scene=scene,
            outdir=Path(scratch) / smoothing,
            name=f"crossfit_template_matching_k3_{smoothing}",
            title=f"Cross-fit AP template matching | sub-{subject} | {smoothing}",
            left_labels=combos[(smoothing, "L")]["label_gifti"],
            right_labels=combos[(smoothing, "R")]["label_gifti"],
            left_surface=combos[(smoothing, "L")]["surface"],
            right_surface=combos[(smoothing, "R")]["surface"],
            spec_path=combos[(smoothing, "L")]["spec_path"],
        )
compose_stacked_rows(subject=subject, rows=rows, output_path=output_path)
```

- [ ] **Step 4: Implement bounded batch execution and provenance artifacts**

```python
jobs = max(1, min(int(jobs), 10, len(tasks) if tasks else 1))
rows = run_render_tasks(tasks, jobs=jobs)
write_status_tsv(output_dir / "individual_overview_status.tsv", rows)
write_json(output_dir / "individual_overview_manifest.json", {"summary": summary, "items": rows})
```

- [ ] **Step 5: Run the new focused test and make it pass**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/individual_workflow/test_crossfit_individual_overviews.py -q
```

Expected: all new tests pass with no actual Workbench invocation.

### Task 3: Document and exercise the production workflow

**Files:**
- Modify: `scripts/individual_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/crossfit_template_matching.md`
- Test: `tests/individual_workflow/test_crossfit_individual_overviews.py`

**Interfaces:**
- Consumes: the new renderer CLI and canonical output resolver.
- Produces: an executable production command, named output tree, and clear description that each image has 2 mm and 4 mm rows.

- [ ] **Step 1: Add the documented production command**

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py --jobs 4 --resume
```

- [ ] **Step 2: Add the documented output contract**

```text
outputs_hcp/present/overviews/crossfit_template_matching/individual/
  sub-<id>_crossfit_template_matching_k3_overview.png
  individual_overview_status.tsv
  individual_overview_manifest.json
```

- [ ] **Step 3: Run the focused test again after documentation/code changes**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/individual_workflow/test_crossfit_individual_overviews.py -q
```

Expected: PASS.

- [ ] **Step 4: Run a one-subject, two-smoothing production smoke render**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py --subjects 115825 --jobs 1
```

Expected: one non-empty canonical PNG and an `ok` row in `individual_overview_status.tsv`.

- [ ] **Step 5: Generate all 151 subject PNGs with bounded concurrency**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py --jobs 4 --resume
```

Expected: `ok=151`, `failed=0`, and 151 output PNGs in the canonical `individual/` subtree.

- [ ] **Step 6: Do not commit without an explicit user request**

The user requested implementation and outputs, not a Git commit. Leave the verified working-tree changes available for review.

## Self-Review

- Spec coverage: two smoothing rows, per-subject PNGs, standard hippocampal visualization, canonical output path, bounded concurrency, provenance, and documentation each have a dedicated task.
- Placeholder scan: no `TODO`/`TBD` steps remain; every test and runtime command is explicit.
- Type consistency: `render_subject_overview` consumes validated `Path` records and returns serializable row metadata; `render_individual_overviews` writes the aggregate artifacts consumed by the documented command.

## Execution Handoff

The direct user request authorizes inline implementation in this session. Execute Tasks 1–3 with test-first checkpoints; do not create a commit unless asked.

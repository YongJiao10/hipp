# Cross-Fit Fisher-z Network Labels Visualization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the presentation-only anterior/middle/posterior legend labels in the canonical individual cross-fit overviews with the Fisher-z identified networks VAN, SMN, and DMN, using the project's canonical network colors.

**Architecture:** Keep cross-fit label codes and source manifests unchanged: they remain longitudinal-axis ordered codes 1/2/3. The overview renderer will fingerprint-validate every fold/smoothing/hemisphere template profile, aggregate each code's 400-ROI Fisher-z profile by the canonical Kong2022 network registry, require a stable one-to-one winner across all templates, and create temporary display GIFTIs whose labels/colors are VAN/SMN/DMN. The persistent overview manifest records this visual-label contract, and a guarded refresh flag permits atomic replacement of legacy AP-labelled PNGs.

**Tech Stack:** Python 3.14, NumPy, nibabel, Pillow, existing Kong2022/network-color helpers, Connectome Workbench via `render_locked_grid`, pytest.

## Evidence locked before implementation

- Source label code `1` has the highest mean Fisher-z profile in `VentralAttention` for all 20 templates (5 folds × 2 smoothings × L/R).
- Source label code `2` has the highest mean Fisher-z profile in `Somatomotor` for all 20 templates.
- Source label code `3` has the highest mean Fisher-z profile in `Default` for all 20 templates.
- Display aliases are `VAN`, `SMN`, and `DMN`; colors are indexed by canonical Kong2022 style keys in `config/hipp_network_style.json`: VentralAttention/VAN `(220,92,255,255)`, Somatomotor/SMN `(99,150,194,255)`, Default/DMN `(216,92,107,255)`.

## File Structure

- Modify: `scripts/individual_workflow/02_01_render_hcp_crossfit_individual_overviews.py` — derive/validate the Fisher-z network mapping, generate temporary network-labelled GIFTIs, record the visual contract, and add guarded refresh behavior.
- Modify: `tests/individual_workflow/test_crossfit_individual_overviews.py` — add failing coverage for the stable mapping, standardized colors, temporary label-table rendering, and stale visual-contract handling.
- Modify: `scripts/individual_workflow/README.md` — state the network-labelled legend and refresh command.
- Modify: `docs/hipp_parcellation_network/crossfit_template_matching.md` — distinguish immutable AP source codes from Fisher-z network display labels and record the corresponding colors/evidence.

## Task 1: Write and run failing behavioral tests

- [ ] Add a synthetic template-profile fixture with three canonical network winners and assert the mapping is `1→VAN`, `2→SMN`, `3→DMN` only when every template agrees.
- [ ] Assert that generated display GIFTI label tables retain native codes but expose names/colors `VAN/SMN/DMN` from the shared style configuration.
- [ ] Assert `--resume` reuses an output only with an identical visual-label contract; an existing legacy AP-only manifest must require explicit visual refresh.
- [ ] Run `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/individual_workflow/test_crossfit_individual_overviews.py -q` and confirm RED failures are about the missing contract/mapping implementation.

## Task 2: Implement provenance-backed display labels

- [ ] Load canonical Kong2022 ROI order through `lib.kong2022_label_utils.load_kong2022_bundle` and use the 400 ROI columns of each fingerprint-validated `template_profiles.npy`.
- [ ] Verify all 5 fold × 2 smoothing × 2 hemisphere profile artifacts, calculate per-network mean Fisher-z values, and reject ties/inconsistent top networks.
- [ ] Resolve display aliases and exact colors through `lib.network_colors.require_network_colors`, with the `Somatomotor→SMN` presentation alias explicit in code.
- [ ] Create scratch-only GIFTIs using source int32 labels plus the new label table; do not mutate cross-fit source labels or their AP provenance.
- [ ] Include the evidence, profile fingerprints, mapping, and color contract in each result row and in the aggregate manifest.
- [ ] Add `--refresh-visuals`, which can atomically replace a validated existing PNG only when the caller explicitly requests refresh; ordinary `--resume` remains non-destructive reuse-only.

## Task 3: Document, validate, and render

- [ ] Update both workflow documents with the source-vs-display distinction and the documented refresh command.
- [ ] Run the focused test file and the relevant cross-fit robustness tests until GREEN.
- [ ] Run a one-subject refresh smoke render, inspect the PNG legend visually, and verify its manifest contract.
- [ ] Refresh all 151 canonical subject images with no more than four worker processes and verify `ok=151`, `failed=0`, all image fingerprints, and a matching visual contract.

## Completion criteria

- Every per-subject PNG has two fixed rows (2 mm then 4 mm) and legends named/color-coded `VAN`, `SMN`, `DMN`.
- The immutable scientific source labels remain AP-ordered in cross-fit manifests; only the display layer changes.
- Fisher-z mapping evidence is machine-readable and reproducible from validated templates.
- Documentation, code, tests, output manifests, and rendered files are synchronized; no commit or deletion occurs without explicit user authorization.

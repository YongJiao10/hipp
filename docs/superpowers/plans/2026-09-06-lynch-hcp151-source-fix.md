# Lynch HCP 151-subject source-cohort repair

## Goal

Remove the hard-coded `nsi >= 0.43` / expected-78 gate from the Lynch HCP source manifest step (`02_03`). The Lynch HCP branch must use the authoritative 151-subject HCP QC-pass cohort, while retaining NSI values as audit metadata. Existing 78-subject artifacts remain untouched as historical outputs. The established renderers, visual layout, and Workbench hippocampal rendering stay unchanged; only the source cohort and downstream data products are refreshed.

## Design

- Make `02_03_build_lynch2024_nsi_manifest.py` pass every subject in the base HCP QC manifest. Do not apply an NSI threshold, expected-count assertion, or missing-NSI exclusion. Write a 151-subject pass manifest, an NSI audit table, and a run manifest that records `selection_rule=all_base_subjects`.
- Give the new shared-input and group-atlas products provenance-specific names so the repaired 151-subject branch cannot overwrite the old `nsi043_78` products.
- Preserve the historical 78-subject contract in compatibility code and paths where old artifacts are still referenced; add a separate 151-subject contract for the repaired branch.
- Point only the HCP Lynch comparison resolver at the new 151-subject roots. Leave Iowa Lynch paths and the six original visualization renderers structurally unchanged.

## Implementation tasks

1. Update `tests/core_workflow/test_lynch2024_nsi_manifest.py` first so it asserts that all base subjects pass, NSI values are recorded in an audit table, and no threshold/count gate is required. Run the focused test to capture the expected failure against the current implementation.
2. Implement the minimal source change in `scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py`, including new 151-subject filenames and manifest metadata. Update its direct documentation and source-workflow docs in the same change.
3. Update `scripts/core_workflow/lib/output_layout.py`, `scripts/core_workflow/lib/formal_cohorts.py`, the valid-mask/input route, and the visualization comparison resolver so the repaired HCP branch resolves to the new 151-subject roots while legacy 78-subject paths remain readable.
4. Rebuild the source products in dependency order with the canonical `py314` interpreter: create the 151-subject manifest, generate missing cortical PFM inputs for the 73 previously excluded subjects, derive 151-subject valid masks, build Lynch20/Lynch7 references, materialize FC, compute gradients, build the Lynch7 group atlas, and regenerate the 151-subject overview. Run no more than one local workflow batch at a time and use the repository’s existing Workbench commands and output conventions.
5. Run the existing HCP/JJTMS 7-network comparison suite through the original renderer modules, verify its Lynch HCP label is `N=151`, verify all consumed manifests report 151 subjects, and confirm the Workbench hippocampal image panels are still present. Do not introduce a replacement plotting implementation.

## Verification

- `/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/core_workflow/test_lynch2024_nsi_manifest.py`
- `/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/visualize/test_hcp_jjtms_7net_branch.py`
- `/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py scripts/core_workflow/lib/output_layout.py scripts/core_workflow/lib/formal_cohorts.py scripts/visualize/lib/multidataset_cohorts.py`
- Manifest/count and output-integrity checks over the repaired 151-subject roots, including Workbench-rendered hippocampal panels and unchanged renderer/layout code paths.

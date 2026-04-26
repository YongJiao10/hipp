# Kong2022 Route A — Individualized PFM (ROI-level) Implementation Plan

## Context

Route B (public-atlas extraction) is fully implemented: every subject gets per-vertex ROI labels from the *group* `Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`, then ROI/canonical-8 timeseries, hippocampal vertex-to-ROI FC, and the group atlas. Route A — *individualized* 400-ROI parcellation per subject before the same downstream chain — has only a stub (`scripts/cortex/run_cortex_pfm_subject_kong2022.py` raises `NotImplementedError`). The existing `fast_pfm` cortex PFM only operates at network granularity (K=17/20/14 priors).

Key enabler discovered during exploration: `external/Schaefer2018_Kong2022/Params_Final.mat` already contains a `theta` matrix of shape `(64984, 400)` — a per-vertex × per-ROI prior, identical in shape (after transpose) to the `(K, V)` priors `FASTANS.fast_pfm` already consumes for `Kong2019_priors.pickle` `(17, 64984)`. The `fast_pfm` math (project → z-score → argmax) is granularity-agnostic; only its hardcoded post-processing (50 mm² min cluster + 50 mm dilation) is network-tuned and needs replacement at ROI scale.

## Scope and Decisions (locked with user)

1. **Algorithm** — fast_pfm WTA at ROI granularity, with ROI-tuned cleanup. Inline the WTA math in our repo (do not patch `external/FASTANS`); replace the network-scale cleanup with a per-parcel keep-largest-component step.
2. **Atlas slug** — `kong2022_pfm` (sibling to `kong2022`). Same canonical-8 mapping; same 400 ID space; same parcel names. Outputs land at `outputs_migration/hipp_group_inputs_shared/sub-{sid}/{reference,fc}/kong2022_pfm/`.
3. **Group driver** — parameterize the hardcoded `"kong2022"` literals in `run_merged_canonical_group_atlas.py` so any Schaefer-400 family atlas works. Per-subject TSV stays the source of truth for the ROI registry; the strict equality check against the public registry is relaxed to per-id name lookup.

## Recommended Approach

### Pipeline shape (mirrors Route B; only step 1 differs)

```
prepare (T1only) ──► kong2022_pfm-reference ──► kong2022_pfm-fc ──► intrinsic-fc ──► group-atlas ──► present
                     │                          │                                     │              │
                     ├ NEW: PFM solver          ├ existing materializer,              ├ atlas slug   └ atlas slug
                     │   + extractor             │   parameterized by --atlas-slug    │  kong2022_pfm  kong2022_pfm
```

Shared with Route B: `prepare` (Dice + hippocampal tSNR), `intrinsic-fc`, the group atlas/present scripts (after parameterization).

### Step-by-step changes

#### 1. Build Kong2022 ROI prior assets (one-time, deterministic)

- Add a small loader helper in `scripts/cortex/kong2022_label_utils.py`:
  - `load_kong2022_roi_priors(params_mat: Path) -> np.ndarray` — reads `Params_Final.mat`, returns `theta.T` cast to float32, shape `(400, 64984)`. Verify dtype/shape and the first/last 5 columns sum to a positive number per vertex.
  - `load_kong2022_label_list(info_txt: Path) -> Path` — already-formatted `Schaefer2018_400Parcels_Kong2022_17Networks_order_info.txt` is the wb_command label-list format. Function returns the path verbatim (rename optional, no parsing needed). Used by `wb_command -cifti-label-import`.
- No on-disk pre-bake of a `.pickle` — `theta.T` is loaded fresh per subject (it is 400 × 64984 × 4 B ≈ 100 MB; cheap).
- No edits to `external/FASTANS/`.

#### 2. Replace stub with real Route A subject driver

`scripts/cortex/run_cortex_pfm_subject_kong2022.py` (replace 71-line stub):

- CLI surface mirrors `run_cortex_pfm_subject.py` (`--subject`, `--dtseries`, `--{left,right}-{midthickness,inflated}`, `--sulc-dscalar`, `--out-root`); plus `--kong2022-params` (default `external/Schaefer2018_Kong2022/Params_Final.mat`), `--kong2022-label-list` (default `Schaefer2018_400Parcels_Kong2022_17Networks_order_info.txt`), `--min-component-area-mm2` (default 5.0), `--keep-largest-only` (default True).
- Output dir: `outputs_migration/cortex_pfm/sub-{sid}/kong2022_pfm/`.
- Algorithm (Python; no FASTANS branching, no `external` patches):
  1. Load `theta.T` via `load_kong2022_roi_priors()` → `priors` `(400, 64984)`.
  2. Load CIFTI dtseries → `(T, 64984)`. Z-score per vertex along T (NaN-safe; mask zero-variance vertices to medial wall).
  3. `prior_ts = priors @ ts.T` → `(400, T)`. Z-score per-row.
  4. `weights = prior_ts @ ts / (T-1)` → `(400, 64984)`.
  5. `labels = argmax_axis0(weights) + 1` → `(64984,)` int32.
  6. Re-mask to template cortex (zero medial wall) using `external/FASTANS/resources/templates/CORTEX.32k_fs_LR.dscalar.nii` template (read once with `nibabel`; do not import FASTANS).
  7. Write `PFM_Kong2022priors.dscalar.nii` then `wb_command -cifti-label-import` with the 400-row label-list to produce `PFM_Kong2022priors.dlabel.nii`.
  8. **ROI-tuned cleanup** (instead of fast_pfm's `50 50` cluster+dilate):
     - For each parcel id 1..400 present in `labels`, isolate that parcel into a binary dscalar.
     - `wb_command -cifti-find-clusters … 0 <min-area> 0 <min-area> COLUMN -merged-volume -corrected-areas <vertex_areas>` to drop islands smaller than `--min-component-area-mm2`.
     - If `--keep-largest-only`, retain only the largest connected component per parcel.
     - Re-import the cleaned dscalar to dlabel using the same label list. Vertex areas come from `wb_command -surface-vertex-areas` (mirrors `derive_cortex_roi_components.py:111-130`).
     - Skip the network-scale `-cifti-dilate 50 50`.
  9. `wb_command -cifti-separate` → `PFM_Kong2022priors.{L,R}.label.gii` (reuse `separate_labels()` from `run_cortex_pfm_subject.py` — refactor it to a shared helper or import it from the same module).
  10. Write `pfm_summary.json` with: subject, params_path, priors_shape, expected K=400, observed K (parcels with ≥1 vertex after cleanup), per-parcel kept vertex counts, n_lost_to_cleanup, label-list source.
  11. Render the same Workbench overview PNG as Route B kong2019 PFM (reuse `render_method()` from `run_cortex_pfm_subject.py`; new method slug `kong2022_pfm`).

#### 3. Route A batch driver

New `scripts/cortex/run_cortex_kong2022_pfm_batch.py`:

- Mirror `run_cortex_pfm_batch.py`. Subjects from `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` (or `--subjects-manifest`).
- For each subject: invoke `run_cortex_pfm_subject_kong2022.py` then immediately invoke the new PFM extractor (step 4) — keeps the "individualization → reference extraction" pair atomic.
- `MAX_CONCURRENCY = 10` (CLAUDE.md hard cap). `ThreadPoolExecutor`.
- Resume gate: subject is "done" when `outputs_migration/hipp_group_inputs_shared/sub-{sid}/reference/kong2022_pfm/{reference_summary.json,stage_manifest.json}` both exist.

#### 4. Per-subject PFM-aware reference extractor

New `scripts/cortex/extract_cortex_kong2022_pfm_roi_timeseries.py` — clone of `extract_cortex_kong2022_roi_timeseries.py` with three deltas:

- `ATLAS_SLUG = "kong2022_pfm"` (drives `reference/kong2022_pfm/` output path).
- Default `--kong2022-dlabel` resolves to the *per-subject* PFM dlabel: `outputs_migration/cortex_pfm/sub-{sid}/kong2022_pfm/PFM_Kong2022priors.dlabel.nii`. Override allowed for ad-hoc reuse.
- The `load_kong2022_bundle(dlabel_path=...)` call still works unchanged because `kong2022_label_utils.py` already accepts an arbitrary dlabel path and parses any 400-row file with the canonical naming convention. The 400-parcel assertion at `kong2022_label_utils.py:110` stays — if a subject's PFM lost a parcel after cleanup it must be reported and abort (or relax to ≥395 with explicit warning + kept rows reported in `reference_summary.json`; recommend abort+manual review for v1).
- Output schema is identical to Route B (`cortex_roi_parcel_timeseries.npy` `(N≤400, T)`, `cortex_canonical_network_timeseries.npy` `(8, T)`, `cortex_roi_parcels.tsv`, `reference_summary.json`, `stage_manifest.json`). The `reference/common/` files (tSNR mask, run boundaries) are **shared** with Route B — if they already exist from a Route B run, accept them via `--{left,right}-tsnr-mask` and skip recomputation; otherwise compute identically.

#### 5. Refactor the FC materializer to be slug-parameterized

Edit `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py` (instead of cloning):

- Add `--atlas-slug` CLI flag (default `kong2022`).
- Replace the module-level `ATLAS = "kong2022"` constant with a function arg threaded through `materialize_fc()` and the `fc_definition` strings.
- Update `materialize_kong2022_vertex_to_roi_fc.py:170,184` to use the slug variable.
- No schema changes; just enables `python … --atlas-slug kong2022_pfm` to write to `fc/kong2022_pfm/...`.

#### 6. Group atlas driver — parameterize Kong2022-specific paths

Edit `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`:

- Line 48: extend `ATLASES = ["kong2019", "kong2019_8net", "kong2022", "kong2022_pfm"]`.
- Add a constant `KONG2022_FAMILY = {"kong2022", "kong2022_pfm"}` near the top.
- `cluster_roi_profile()` (lines 937–1009): add `atlas: str` parameter; replace literals at 955–956 with f-string templates `reference/{atlas}/cortex_roi_parcel_timeseries.npy` and `reference/{atlas}/cortex_roi_parcels.tsv`.
- Lines 985–1003 (strict per-subject registry equality vs the public bundle): keep the per-id `parcel_id ∈ public 400` check, but downgrade name/family/canonical8 mismatches from `RuntimeError` to a single warning with `(subject, roi_id, observed, expected)` rows so individualized subjects with renamed-but-id-stable parcels still flow through. Recommend: keep `RuntimeError` as the default and add `--allow-registry-name-drift` flag for Route A; set the flag in the Route A pipeline call.
- Lines 1223 (partial-method skip), 1456 and 1472 (ROI-profile branch): change `if atlas == "kong2022"` → `if atlas in KONG2022_FAMILY`.
- Line 1457: pass `atlas=atlas` through to `cluster_roi_profile`.
- Loader `load_kong2022_roi_registry()` (line 899) stays — it returns the canonical 400-row registry which is a valid label space for both atlases.
- The unsupported-method (`partial-spectral`) skip still applies to Route A (same coverage logic).

Edit `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py:34`:
- Add `"kong2022_pfm"` to its `ATLASES` list.

#### 7. Config: register the new atlas slug

Edit `config/cross_atlas_network_merge.json` — append a sibling block under `atlases`:

```json
"kong2022_pfm": {
  "display_name": "Kong2022_PFM",
  "original_network_count": 17,
  "merged_functional_network_count": 8,
  "roi_count": 400,
  "source_dlabel_relpath": "external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii",
  "source_info_relpath": "external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order_info.txt",
  "params_relpath": "external/Schaefer2018_Kong2022/Params_Final.mat",
  "mapping": { "...same family17→canonical8 mapping as kong2022..." }
}
```

Update `kong2022_label_utils.load_kong2022_merge_mapping()` (line 81) to accept an `atlas_slug` arg (default `"kong2022"`), threaded through `load_kong2022_bundle()`. Route A passes `"kong2022_pfm"`. The two mappings are identical for now but the indirection lets them diverge cleanly.

#### 8. T1only pipeline driver for Route A

New `scripts/hipp_parcellation_network/run_t1only_kong2022_route_a_pipeline.py` — clone of `run_t1only_kong2022_route_b_pipeline.py` with stage list:

```python
STAGES = ["prepare", "kong2022_pfm-individualize", "kong2022_pfm-reference",
          "kong2022_pfm-fc", "intrinsic-fc", "group-atlas", "present"]
```

Per-stage subprocess invocations:
- `prepare` → reuse Route B's `prepare_t1only_group_inputs.py`.
- `kong2022_pfm-individualize` → `scripts/cortex/run_cortex_kong2022_pfm_batch.py` (PFM solver only).
- `kong2022_pfm-reference` → embedded in the batch driver above (one-pass), or split as a second stage that just calls the new extractor for resume granularity. Recommend split.
- `kong2022_pfm-fc` → `materialize_kong2022_vertex_to_roi_fc.py --atlas-slug kong2022_pfm`.
- `intrinsic-fc` → reuse Route B (atlas-invariant).
- `group-atlas` → `run_merged_canonical_group_atlas.py --atlases kong2022_pfm --methods group-network-spectral group-intrinsic-spectral --coverage-thresholds 0.80 --allow-registry-name-drift`.
- `present` → `build_present_group_overview_merged_canonical.py --atlases kong2022_pfm …`.

`--from-step` / `--until-step` semantics preserved. `HIPP_MAX_CONCURRENCY=10`. Logs at `<route_root>/logs/pipeline_<utc>/<stage>.log`.

#### 9. Documentation

- New `docs/hipp_parcellation_network/kong2022_route_a_flow.md` — mirror the structure of `kong2022_route_b_flow.md`. Sections: Summary (PFM-based individualization), Pipeline Steps, Upstream Common Inputs (shared with Route B), Group Outputs Specific to kong2022_pfm, Method Support, Differences vs Route B (one-paragraph table comparing the two routes' first stage).
- Update `docs/hipp_parcellation_network/merged_canonical_group_atlas_plan.md` — add `kong2022_pfm` to the atlas list, document the parameterization of `cluster_roi_profile`, and note the `--allow-registry-name-drift` flag.
- Update `external/Schaefer2018_Kong2022/README.md` — change "Route A — stub only" to a link to the new flow doc and runbook; document `Params_Final.mat["theta"]` shape.
- New runbook `docs/hipp_parcellation_network/t1only_kong2022_route_a_runbook.md` — clone of the Route B runbook with the stage-list change and the recommended flag set.
- Per CLAUDE.md "Doc Sync Rule" and "Flow Doc Maintenance" — update `docs/hipp_parcellation_network/multi_branch_flow.md` and `step_by_step_flows.md` to mention the Route A `kong2022_pfm` branch in the network-prior arm.

## Critical files to modify or create

| Path | Action |
|---|---|
| `scripts/cortex/run_cortex_pfm_subject_kong2022.py` | **Replace stub** with real PFM solver (≈220 lines). |
| `scripts/cortex/extract_cortex_kong2022_pfm_roi_timeseries.py` | **New**, clone-and-modify of `extract_cortex_kong2022_roi_timeseries.py`. |
| `scripts/cortex/run_cortex_kong2022_pfm_batch.py` | **New**, batch driver with concurrency cap 10. |
| `scripts/cortex/kong2022_label_utils.py` | Add `load_kong2022_roi_priors()`, thread `atlas_slug` through `load_kong2022_merge_mapping()` and `load_kong2022_bundle()`. |
| `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py` | Add `--atlas-slug` flag; thread slug through `materialize_fc()`; remove module-level `ATLAS` constant. |
| `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py` | Extend `ATLASES`; introduce `KONG2022_FAMILY` set; parameterize `cluster_roi_profile(atlas=...)`; replace literals at lines 955–956; widen guards at 1223/1456/1472; add `--allow-registry-name-drift` flag (default False). |
| `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py` | Extend `ATLASES`. |
| `scripts/hipp_parcellation_network/run_t1only_kong2022_route_a_pipeline.py` | **New**, clone Route B with new stage list. |
| `config/cross_atlas_network_merge.json` | Add `atlases.kong2022_pfm` block. |
| `external/Schaefer2018_Kong2022/README.md` | Update Route A section. |
| `docs/hipp_parcellation_network/kong2022_route_a_flow.md` | **New** flow doc. |
| `docs/hipp_parcellation_network/t1only_kong2022_route_a_runbook.md` | **New** runbook. |
| `docs/hipp_parcellation_network/merged_canonical_group_atlas_plan.md` | Add Route A section; document parameterization. |
| `docs/hipp_parcellation_network/multi_branch_flow.md`, `step_by_step_flows.md` | Reference Route A in the network-prior arm. |

## Reused functions / utilities

- `kong2022_label_utils.load_kong2022_bundle()` — already accepts arbitrary dlabel path; works unchanged for per-subject PFM dlabel that preserves Schaefer naming convention.
- `kong2022_label_utils.split_vertex_label_arrays()`, `build_roi_registry()`, `parse_kong2022_roi_name()` — reuse verbatim.
- `extract_cortex_kong2022_roi_timeseries.{compute_cortex_tsnr_gate, ensure_vertices_aligned, parse_run_segments, aggregate_hemi_roi_timeseries}` — clone-and-extend in the new PFM extractor (or import directly; recommend import to share the tSNR / run-boundary logic exactly).
- `extract_cortex_merged_network_timeseries.{extract_structure_data, load_required_mask}` — reuse verbatim.
- `run_cortex_pfm_subject.{separate_labels, separate_sulc, render_method, run, WB_COMMAND, REPO_ROOT}` — reuse directly. **Do not** call `import_fastans` (Route A inlines the math).
- `derive_cortex_roi_components.py` cluster-finding pattern (`wb_command -metric-find-clusters` with `-corrected-areas`) — adapt for the per-parcel ROI cleanup at the dscalar level.
- `materialize_kong2022_vertex_to_roi_fc.py` after parameterization — reused for both atlases.
- `run_merged_canonical_group_atlas.py` (after parameterization) and `build_present_group_overview_merged_canonical.py` — reused for both atlases.
- T1only `prepare`, `intrinsic-fc` stages — reused as-is from Route B.

## Verification plan

End-to-end smoke test on **3 subjects** from `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`:

1. **Solver smoke** — run `run_cortex_pfm_subject_kong2022.py` for one subject:
   - `PFM_Kong2022priors.dlabel.nii` exists; `wb_command -file-information` reports 400 unique non-zero label IDs.
   - `pfm_summary.json` records observed K and per-parcel kept vertex counts.
   - Rendered overview PNG: visually confirm spatially-coherent parcels (no salt-and-pepper).

2. **Extractor smoke** — run `extract_cortex_kong2022_pfm_roi_timeseries.py`:
   - `reference/kong2022_pfm/cortex_roi_parcel_timeseries.npy` shape `(N, 3600)`, N close to 400.
   - `cortex_canonical_network_timeseries.npy` shape `(8, 3600)`.
   - Compare canonical-8 Pearson r vs Route B same subject: expect `r > 0.85` per network.

3. **FC smoke** — run `materialize_kong2022_vertex_to_roi_fc.py --atlas-slug kong2022_pfm`:
   - `fc/kong2022_pfm/2mm/hemi_L/hipp_vertex_to_network_fc.npy` shape `(n_hipp_vertices, N_rois)`.

4. **Group atlas smoke** — run on 3-subject set:
   ```
   python scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py \
     --atlases kong2022_pfm --methods group-network-spectral \
     --smoothings 2mm --hemis L \
     --coverage-thresholds 0.80 --subjects 100610 102311 104416 \
     --allow-registry-name-drift
   ```
   Verify `cluster_roi_profile.npy` shape `(K_final, 400)` and `cluster_roi_top20.json` produced.

5. **Side-by-side check** — render `present_group/kong2022_pfm_group-network-spectral_group_overview.png`; compare against existing `kong2022` overview.

6. **Full cohort run** — after smoke test passes: T1only Route A pipeline, all 153 subjects, `--jobs 8`.

7. **Doc/sync verification** — `multi_branch_flow.md`, `step_by_step_flows.md` mention Route A; `README.md` no longer says "stub only".

## Risks and mitigations

- **Cleanup destroys parcels.** Small Schaefer parcels could lose all area. Mitigation: tune `--min-component-area-mm2` down to 1.0; report `n_lost` in `pfm_summary.json`; abort extractor if `n_lost > 5`.
- **Zero-variance vertices.** `std=0` after motion scrubbing → NaN in standardized ts. Mitigation: replace NaN with 0 (vertex contributes 0 to WTA score; re-masked by medial-wall template after argmax).
- **`theta` semantics.** Treated as per-vertex × per-ROI weight; argmax is valid regardless of scale/sign. Mitigation: report per-vertex `argmax(theta)` overlap with the public dlabel in smoke test (expect ≥80%).
- **Strict registry check.** Per-id assertion `parcel_id ∈ {1..400}` must hold. Mitigation: extractor enforces 400-parcel completeness or aborts before group driver runs.
- **Storage doubling.** Second `reference/kong2022_pfm/` + `fc/kong2022_pfm/` tree per subject. Not blocking; document in runbook.

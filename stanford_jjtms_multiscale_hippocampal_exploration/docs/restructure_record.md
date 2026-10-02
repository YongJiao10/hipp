# Restructure record

## 2026-09-01 — Task 1

The evidence-based method decision is frozen before later implementation work:

- `area_source = hippunfold_native_surfareacorobl` is the sole written decision
  for the Kember-aligned area endpoint.
- Absolute native surface area in mm² is primary. The explicitly defined surface
  area fraction is secondary context.
- `compute_vertex_area` remains geometric-only, and `transition_area` is not a
  Kember endpoint.
- `cross_dataset_rest_fc`, `kember_metrics`, and
  `external_group_atlas_analysis` remained separate workflow surfaces at that
  checkpoint, with separate estimands and forbidden cross-usage rules. The
  group-atlas scheduling boundary was superseded by the 2026-09-02 correction
  recorded below.

The contract is recorded in [kember_area_method.md](kember_area_method.md) and
[analysis_map.md](analysis_map.md). The existing README, literature notes, and
study specification point to this contract. The document-status tests protect
the method name, endpoint exclusions, geometric-only distinction, and the exact
three-workflow map.

## 2026-09-01 — Task 2: canonical workflow

The duplicated subject-level branches were consolidated before output cleanup:

- `kember_area/` was renamed to `kember_metrics/` and is now the only Kember package.
- `fast_rest/` code was moved or retired according to workflow role; it is not an active
  Kember input or output path.
- The active Kember area operation consumes native HippUnfold `surfareacorobl` values;
  geometry reconstruction is limited to a clearly named QC/utility role.
- The active subject-level method axis contains only
  `hcp151_template_matching` and `hcp151_fixed`.

Stable code paths and their documentation now agree on one overwrite-in-place layout:

- `kember_metrics/work/features/`
- `kember_metrics/results/direct_contrasts/`
- `kember_metrics/figures/direct_contrasts/`
- `kember_metrics/reports/direct_contrasts/`

## 2026-09-01 — Rebuilt outputs

The feature and direct-contrast outputs were rebuilt from the current configuration:

- 1,552 subject/method/smoothing/hemisphere/system combinations completed successfully.
- 4,656 feature rows all use native HippUnfold `surfareacorobl` with metric paths and input hashes.
- 432 direct-effect units are present for six contrasts, two methods, two smoothings,
  two hemispheres, three metrics, and three systems.
- The stable report records zero duplicate feature/effect keys, zero missing coverage cells,
  and finite raw p values for all 432 effect rows.

## 2026-09-01 — Atlas field namespace

At the 2026-09-01 checkpoint, the external seven-cohort group atlas remained a
separate group-level geometry workflow. Its area fields were renamed and
regenerated so that they cannot be confused with Kember metrics; current
scheduling is superseded by the 2026-09-02 unified branch-ledger correction:

- `atlas_region_area_mm2` is the triangle-area/3 atlas geometry sum by cluster.
- `atlas_region_area_fraction` is the corresponding atlas-only fraction.
- `atlas_total_surface_area_mm2` is the atlas mesh total.
- These fields are descriptive group-atlas outputs and never subject-level Kember area.

## Cleanup record

The approved cleanup scope removes the obsolete package, pilot, repeated run trees and
superseded plan/spec documents that would otherwise compete with the current workflow:

- legacy `fast_rest/` package and its results, reports, figures, and work files;
- the relocated pre-rebuild native-area pilot under `kember_metrics/results/`;
- `primary_screen_v1*` attempt result trees and `core_results_network_localisation_v2/v3/v4` figures;
- the obsolete top-level implementation/specification documents and `branch_spec_corrections.md`.

The deletion is restricted to this project directory and is recorded again after execution.

## 2026-09-01 — Cleanup executed

After the final path-resolution check and user confirmation, the approved cleanup was executed:

- the legacy package and its old result/report/figure/work tree were removed;
- the relocated pre-rebuild pilot and four repeated `primary_screen` attempt trees were removed;
- the six obsolete v2/v3/v4 localisation figure files were removed;
- the obsolete implementation/specification documents were removed.

All confirmed cleanup targets are absent. The stable unversioned workflow outputs and the
renamed current package are now the only active paths.

## 2026-09-01 — Group atlas scope correction

The external atlas package was renamed from the misleading
`external_four_group_atlas_analysis/` path to
`external_group_atlas_analysis/` and generalized to the seven configured cohort maps:

- parent factorial groups: NTHC, TEHC, NTS, and TIS;
- nested clinical cohorts: NTS Any MDD, NTS Pure MDD, and TIS PTSD;
- 56 validated cells across cohort × assignment variant × smoothing × hemisphere;
- 21 descriptive cohort pairs plus explicit nested-to-NTHC summaries;

## 2026-09-01 — Area figure retention correction

The 2 mm comprehensive area figure was incorrectly removed with the retired
`fast_rest/` tree. It is retained at
`kember_metrics/figures/retained_exploratory/area_2mm_comprehensive_summary.png`
and its PDF companion, with provenance recorded beside the files. This figure
is deliberately separated from the active Kember result because its historical
area values used `consumed_midthickness_triangle_area`; the active endpoint
uses native HippUnfold `surfareacorobl`. The complete figure is preserved for
raw subject-level distribution context and is not mixed into current result
tables.

## 2026-09-01 — Current comprehensive area figure

The deleted figure now has a maintained current generator at
`kember_metrics/src/plot_area_2mm_comprehensive_summary.py`.
It consumes the active native Kember area table, expands the seven configured
cohort rosters including `TIS PTSD` (35 subjects), and writes only
`kember_metrics/figures/direct_contrasts/area_2mm_comprehensive_summary.png`.
No PDF is generated or overwritten by this entry point; the historical PDF
under `figures/retained_exploratory/` is archival only.

## 2026-09-01 — HCP 7T cross-fit reference and layout update

The current comprehensive PNG now keeps the seven Stanford direct contrasts intact and adds a separate HCP 7T reference branch:

- HCP labels come from each subject's family-aware five-fold held-out cross-fit result (`individual_labels_k3.npy`); the full HCP template is never used to score its contributing subject.
- Each 512-source label map is transferred to native space with `ADAP_BARY_AREA`, then summed over native HippUnfold `surfareacorobl` per-vertex area values.
- HCP 7T−NTHC is a descriptive cross-dataset Welch comparison and is excluded from the Stanford direct-contrast FDR family.
- The figure uses explicit GridSpec margins with compact row spacing, a wider left margin, wrapped horizontal cohort labels, an enlarged typography scale, and extra significance-bracket headroom; repeated right-panel forest y-axis labels are suppressed to keep them out of the left plot. The footer is intentionally omitted to reduce dead space, and the area-effect forest uses light-gray horizontal separators between groups instead of an unexplained gray x-grid.
- The generator writes only `area_2mm_comprehensive_summary.png`; no PDF is generated or overwritten. The cache run manifest records a full input-asset SHA fingerprint. The HCP feature cache, HCP 7T−NTHC effect table, and full combination provenance are under `kember_metrics/work/reference_hcp7t_crossfit_area/` and are ignored by Git.

## 2026-09-02 — Unified group-atlas branch-ledger correction

The 2026-09-01 standalone atlas scope is superseded for current execution. The
group-atlas geometry family is now registered inside the unified
`cross_dataset_rest_fc` primary branch ledger and shares its manifest, strict
priority barrier, and report closure with subject-level Rest-FC branches.
The compatibility directory and CLI remain only for diagnostics; they are not an
independent authoritative experiment path.

The current registry uses HC plus the seven configured cohorts (64 atlas cells).
Group-wise outputs contain only the seven non-HC-versus-HC comparisons; the old
21 all-pair output is not part of active execution. Geometry tables remain
separate from subject-level FC rows, and `individual_gradient_k3_full` remains
excluded.

## 2026-09-02 — Expired output archive and completed primary run

After explicit confirmation, 14 superseded generated files were moved to
`_archive/2026-09-02-expired-results/`, preserving their relative directories:

- the orphaned pre-unification `primary_screen/run_manifest.json`;
- the old standalone atlas manifest, five CSV result tables, and six figure files (PNG/PDF at that time; PDF variants are no longer retained),
  and `technical_summary.md` from the former 21-pair/all-pairs scope;
- the active compatibility `README.md` was deliberately retained in place.

The canonical-seed `primary_screen` rerun consumed the repository-root current
inputs and completed the unified experiment:

- 80/80 branches completed with strict P0=36, P1=8, P2=36 ordering;
- active methods are `hcp151_fixed`, `hcp151_template_matching`, and
  `group_atlas_gradient_spectral`; no individual-gradient branch was executed;
- group-atlas outputs contain 64 inventory rows, 192 region-area rows, 56
  non-HC-versus-HC matched-comparison rows, and 64 stability rows;
- static PNG figures, tables, technical Markdown, and SHA-256 manifest are

## 2026-09-03 — Deferred multiplicity correction and PNG-only figure policy

The primary localisation and Kember direct-contrast workflows now retain raw
p-values, confidence intervals, and effect sizes without applying multiplicity
correction. The compatibility q columns are null and explicitly marked
`deferred_until_analysis_set_freeze`; no family denominator or q threshold is
used before the final analysis-object set is frozen. The current saved
primary snapshot was revalidated against the corrected localisation code before
its reports and PNG derivatives were rebuilt.

All active figure writers now emit PNG only and their manifests/reports contain
no PDF output entry. After the user confirmed the exact deletion list, the
four remaining PDF artifacts were removed; the project now contains no PDF
figure files.

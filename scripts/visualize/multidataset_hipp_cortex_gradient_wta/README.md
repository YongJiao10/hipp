# Hippocampal parcellation, cortical FC and Iowa HC contrasts

The canonical seven-network figure contains five cohort rows: HCP 7T (151),
Iowa HC (67), Iowa Patient (26; sub-3003 excluded), Pure MDD (7), and Comorbid
MDD (19). The latter two are subgroups of Patient; the displayed total N=270
sums row memberships, not unique individuals. All four smoothing/preprocessing
combinations are generated: 2mm/4mm × 36P/22P. HCP uses its shared ICA-FIX data
in both preprocessing comparisons; the Iowa HC reference follows each variant.

Each row contains the cohort card, Gradient hippocampal atlas, Anterior/Body/
Posterior cortical projections, WTA hippocampal atlas, and VAN/SMN/LAN/DMN
cortical projections. The two 2mm/36p branch figures
(`multidataset_hipp_cortex_gradient_wta_2mm_36p_standard.png` and
`multidataset_hipp_cortex_gradient_wta_2mm_36p_ctxreg.png`) append a third
template-matching section per row: TM Hippo Atlas plus TM Anterior (VAN) /
TM Middle (SMN) / TM Posterior (DMN) cortical projections (see
"Template-matching dominant-atlas third column" below). Each cortical column stacks:

1. Mean Fisher-z FC, using the existing four Workbench views.
2. Top ten negative and positive ROI bars, colored by canonical7 network.
3. Significant differences for that cohort minus Iowa HC in the same four views, where inner fill indicates canonical 7 functional network identities and bold outlines represent the direction of the t-statistic (Red: positive t / higher than HC; Blue: negative t / lower than HC, uncorrected two-sided p < 0.05). The footer provides a dedicated Red (+) / Blue (−) badge legend in place of a continuous t colorbar.
4. Clean slate-gray dashed horizontal dividers (linestyle `(6, 5)`) separate adjacent cohort rows across the figure.

Neither HCP 7T nor Iowa HC displays a connectivity strength comparison vs Iowa HC.
The first two rows (Row 0: HCP 7T, Row 1: Iowa HC) only display the 1000×1408
composite (Mean FC 4-view surface map + Top 10 ROI bars) without a 3rd tile.
Rows 2, 3, and 4 (Iowa Patient, Pure MDD, Comorbid MDD) retain the 3rd tile
(t vs Iowa HC) stacked to 1000×2186. The layout uses unequal row heights
(height ratios 1408:1408:2186:2186:2186), eliminating empty vertical space while
keeping brain map width and physical resolution identical across all five rows.
The standalone renderer remains disabled; use the suite entrypoint below.

## Statistical definition

Individual FC arrays have shape `(N, 7, 400)`. The first three panels are
Anterior, Body, Posterior in registered head-to-tail order. The four WTA rows
are stored as Default, VentralAttention, Somatomotor, Language and displayed
as VAN, SMN, LAN, DMN. ROI IDs/names are aligned to the canonical 400-ROI registry.

FC is Pearson correlation between the mean time series of a hippocampal
cluster/network and each cortical ROI time series, transformed to Fisher z.
It is not an average of vertex-level correlations. Left and right hippocampal
profiles are aligned by subject ID and averaged within subject; hemispheres
are not additional independent observations.

Each panel uses a two-sided independent-samples Welch t test without
covariates, with positive t indicating cohort > Iowa HC. Significance uses the
uncorrected two-sided per-ROI threshold p < 0.05; no FDR or other multiplicity
correction is applied. Patient and subgroup comparisons are separate,
overlapping comparisons.

Missing files, frame mismatches, partial-frame nonfinite series, duplicate IDs
and ROI mismatches fail closed. Empty support, all-missing ROI and constant
signals remain missing, not zero FC observations. A ROI needs at least two
finite observations in each group and a nondegenerate standard error; otherwise
t/p are NaN when a ROI is untestable. Effective N is recorded per ROI. An
entirely absent WTA network is neutral zero only in the upper mean display,
never in t tests.

## Canonical sources and approved recovery

The active `kong2022_7net` overview and comparison manifests use the
`kong2022` producer namespace internally. `canonical7` is the downstream
taxonomy and `kong2022_7net` is the presentation namespace; it never denotes a
seven-column gradient input. Every active gradient source is an ordered
400-ROI profile (`cluster_roi_profile.npy`, `(3,400)`), and every active WTA
source is a vertex-by-400-ROI profile. The four canonical grids are recorded
in `contrasts/canonical_all_grids.manifest.json` and carry `(N,7,400)`
subject matrices.

Standard subject-level FC extraction uses separate, explicit input roots. HCP
surface time series and masks come from
`outputs_hcp/inputs/kong2022/shared_inputs/surface`, while its Kong2022
cortical reference arrays and registry come from
`outputs_hcp/inputs/kong2022/shared_inputs/pfm`. Iowa uses the matching
`outputs_iowa_jj_tms/inputs/kong2022/shared_inputs{_22p}/surface`
and `.../shared_inputs{_22p}/routea` roots. The `shared_inputs` parent itself
is not a subject-input root; resolving it directly is an error. CtxReg keeps
its signal inputs under the explicit `routea_ctxreg` branch while retaining the
validated standard `routea` reference root where the branch contract requires
it.

Gradient means remain the producer-registered `cluster_roi_profile.npy`, shape
`(3,400)`, with `roi_ap`, `long_axis_head_to_tail`, `[1,2,3]` contracts. Individual
exports must reproduce these means; the renderer never substitutes a raw
cluster label file, network profile, older image or approximation for them.
WTA means are validated `(4,400)` `wta_roi` caches with exact bilateral label,
support, label-map and manifest fingerprints.

The original current-input audit found 36P hippocampal/cortex frame mismatches
for 16 of 26 Iowa patients (4 Pure MDD and 12 Comorbid MDD). Example: sub-3010
has 659 denoised hippocampal frames, while the changed cortical reference has
690 frames. The user explicitly approved the named `original-denoised`
recovery and rebuilding upper WTA mean FC from the recovered individuals.

In this mode only Iowa36P reads original time series and valid masks under
`outputs_iowa_jj_tms/work/hcp151_fixed_template_matching/subjects/sub-*/`:

- `surface/{2mm|4mm}/hemi-{L|R}_timeseries.npy`
- `surface/tsnr_gate/hemi-{L|R}_valid_mask.npy`
- `cortex/kong2022_roi_timeseries.npy`

Every file is hash-verified against the original `subject_manifest.json`, and
frame counts must match its denoised run evidence. HCP and Iowa22P keep their
canonical inputs. No upstream reference file is overwritten or truncated.

The formal ROI registry path is
`outputs_hcp/work/group_atlas/pfm_ctxreg/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/cluster_roi_registry.tsv`.
A migrated registry is reusable only when its SHA-256 matches the
authoritative registry hash recorded in the manifest:
`6063230dde6afdb161ab45871fda25fa3e3df1ca2d36632f9b12058596871cee`.
Only the row-identity table is used; FC and labels require their own
validated provenance and are never accepted merely because a path exists.
IDs/names must also match the current canonical registry.

The old Patient manifests contain a retired `shared_inputs*/kong2022_7net`
reference root and no `cluster_roi_profile_subjects_used`. Explicit recovery
reconstructs inclusion from the complete 26-person roster and checked inputs,
requiring its recomputed Gradient means to reproduce the canonical result.
This applies to the missing Patient metadata in both preprocessing variants;
22P still uses its current canonical time series. A present but inconsistent
inclusion list is always an error. Canonical mode does not relax these checks.

Iowa36P WTA mean caches are rebuilt from the same bilateral individual values
used by the t tests. Existing NPY/manifest files are copied to unique
`.backup-recovery-*` files before replacement. Their new manifests register
recovery mode, input fingerprints, bilateral policy and backup hashes. The
canonical7 group WTA parcellation labels themselves are unchanged.

Historical files called `hipp_vertex_to_network_fc.npy` may contain 512×400
ROI Pearson FC, as their summaries show. Names do not establish granularity;
these files cannot replace cluster-average-time-series FC by averaging rows.

## Signal-branch boundary

The maintained standard branch keeps the established unsuffixed artifact names.
A caller that supplies signal_branch="ctxreg" must also supply explicit
CtxReg gradient/WTA roots and branch-qualified panel/cache paths. Source
manifests must record signal_branch="ctxreg"; the only compatibility exception
is the named HCP Kong manifest contract whose recorded consumed inputs prove
routea_ctxreg signal roots and the standard reference root. Missing, partial,
or cross-branch declarations fail closed. The renderer never substitutes a
standard gradient profile, WTA cache, panel, PNG, manifest, or contrast export.

The lower-level contrast preparation accepts an explicitly selected branch.
The established Standard four-grid receipt keeps the historical
`contrasts/{version}_all_grids.manifest.json` name; a CtxReg-only preparation
uses a branch-qualified name such as
`contrasts/canonical_2mm_36p_ctxreg.manifest.json`, or an explicit caller-
provided manifest path. `load_contrasts` requires the requested grid and
signal branch to be present and will not reuse a Standard receipt for CtxReg.
The suite entrypoint still schedules the maintained four-cell Standard grid;
CtxReg preparation must therefore be an explicit lower-level invocation until
the runner is separately extended. No missing WTA cache is silently rebuilt by
the contrast loader, and the historical `contrast_preflight_errors.txt`
remains diagnostics rather than a production artifact.

## Template-matching dominant-atlas third column (2mm/36p branch figures)

Only the two 2mm/36p branch figures carry the 14-column layout; without an
explicit `--tm-contrast-manifest` the renderer keeps the legacy 10-column
Gradient+WTA layout, so all other grids are byte-unaffected by this addition.

Per row the third section holds one TM Hippo Atlas cell plus three cortex
cells (TM Anterior/VAN, TM Middle/SMN, TM Posterior/DMN). Each TM cortex cell
stacks the same tiles as Gradient/WTA cells: upper Mean Fisher-z FC four-view
map + Top-10 ROI bars (1000x1408), and for Patient/Pure MDD/Comorbid MDD rows
a lower t-vs-Iowa-HC tile with canonical-7 inner fill and Red (+)/Blue (-)
significant-cluster outlines stacked to 1000x2186. Row heights therefore follow
the existing 1408 (HCP 7T, Iowa HC) vs 2186 (clinical rows) rule unchanged.

Masks are each Iowa cohort's own group dominant atlas: vertex-wise majority
vote over the branch `hcp151_fixed_template_matching[/routea_ctxreg]/subjects`
2mm `labels.npy`, with deterministic mean-`template_correlations.npy`
tie-break — the same rule as
`scripts/visualize/template_matching_dominant_subregions`. HCP 7T has no
template-matching atlas, so its entire TM section (hippo + three cortex
cells) is blank; Iowa HC shows upper tiles only, never a self-comparison.

The TM Hippo cell is rendered in the locked biglegend style, pixel-aligned
with the Gradient/WTA hippo cells: the same Workbench scene, group-average
den-512 template surfaces, spec and title scheme, with TM dominant labels
written via `make_network_label_gifti` (Anterior/VAN purple, Middle/SMN blue,
Posterior/DMN coral). The per-hemisphere percentage legend on the right is
computed by the scene from the rendered labels. Panels are cached under
`panels/*hippo_tm_biglegend_2mm/` with full upstream fingerprints (scene,
spec, template surfaces, gradient manifests, TM combo manifests/labels/scores)
and any missing input fails closed. There is no tight4view reuse and no
fallback: the TM hippo cell never substitutes a pre-rendered panel.

FC math, bilateral policy, Welch rule (two-sided, uncorrected p<0.05),
±0.15 display scale and contour language are identical to the canonical
columns. TM reuses the existing network footer; it does not add a second
subregion legend. TM contrasts live in separate receipts so canonical 7-panel products are
untouched: `contrasts/canonical_2mm_36p_tm.manifest.json` (standard) and
`contrasts/canonical_2mm_36p_ctxreg_tm.manifest.json` (ctxreg) with schema
`tm_subject_roi_welch_hc_v1_uncorrected`, panels
`TM-Anterior/TM-Middle/TM-Posterior` and `(N,3,400)` subject matrices plus
`_tm_vs_iowa_hc.npz/.tsv` per clinical cohort. The renderer accepts them via
`--tm-contrast-manifest`; the suite prepares per-branch TM manifests
(standard follows `--iowa36p-source`, ctxreg always `canonical`) and passes
the matching manifest per grid cell. Renderer changes conservatively
invalidate contrast receipts, so the 2026-09-16 TM addition re-derived the
`canonical_standard__ctxreg` two-grid contrasts deterministically from the
same inputs (t peak ±6.0 reproduced); overwritten canonical contrast files
were first copied to `.backup-tm-column-20260916T175810Z` siblings.

## Run and reuse

First build, or an intentional rebuild after resolving input contracts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures hipp_cortex_gradient_wta --rebuild-contrast-profiles --iowa36p-source original-denoised
```

Subsequent refreshes reuse saved individual matrices across all 4 grids in 4-way parallel:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures hipp_cortex_gradient_wta --iowa36p-source original-denoised
```

Missing or stale exports fail with instructions to explicitly rebuild. There
is no existence-based fallback or automatic time-series extraction. Source
hash verification still reads input bytes. Code, source mode, IDs and panel
order are validated; extraction/renderer dependency changes conservatively
invalidate exports. The figure stage reads canonical means and validated
statistics, not time series. Runs are sequential with numerical threads limited
to one on IDAS (cap 10). No new runtime dependency was added.

Unsuffixed PNGs are the canonical outputs under the single canonical version
(`canonical_all_grids.manifest.json`). The suite passes `--contrast-manifest`
internally; direct production rendering is disabled.

CtxReg panel/cache products carry the `_ctxreg` suffix, including bilateral
WTA profile caches and branch-specific contrast receipts. Standard products
retain their established unsuffixed names so existing validated Standard
outputs remain addressable without ambiguity.

## Layout, colors and cache

Both cortex maps are 1000×700 px and share the same Workbench scene/cameras.
The 2-tile FC+bars composite is 1000×1408 (used for Row 0: HCP 7T and Row 1: Iowa HC);
the 3-tile composite with caption area and t tile is 1000×2186 (used for clinical rows
2–4). The five-row figure uses dynamic height ratios [1408, 1408, 2186, 2186, 2186]
and scales to 58×58.5875 inches at 220 DPI (12760 × 12889 px), ensuring the cortical
brain maps in Row 0 and Row 1 have the exact same horizontal width and scale as Rows 2–4
without any wasted vertical padding.

FC uses the original blue/red scale, ±0.15, labeled `Mean Fisher-z FC (upper
maps)`. In the lower group contrast maps, inner fill indicates canonical 7 functional
network identities, and bold outlines represent the direction of the t-statistic
(Red: positive t / higher than HC; Blue: negative t / lower than HC, uncorrected
two-sided p < 0.05). In place of a continuous t colorbar, the footer features a
dedicated Red (+) / Blue (−) badge legend. Horizontal slate-gray dashed dividers
(linestyle `(6, 5)`) separate cohort rows across the figure. The p<0.05 rule is
reported per ROI without a correction family.

Before accepting hippocampal panels, both actual GIFTI label tables and source
manifests must match canonical7 and the cohort N. Auditory is forbidden; only
the seven named networks plus Null/unassigned are allowed.

Workbench panel caches contain source/asset/profile/vertex fingerprints,
render parameters, dimensions and output SHA-256. t composites additionally
register t/p/masks, palette, statistical peak, display range, caption and code.
Invalid panels are regenerated from validated inputs, never substituted as data
sources.

Two files named
`panels/iowa_pat_36p_wta_pseudocluster_bilat_{2mm,4mm}.manifest.json` are
retained historical four-network pseudo-cluster records. They are excluded from
the active canonical suite and are never accepted as canonical inputs;
their old provenance is kept only for traceability. The active suite uses the
canonical `routea_patient*` group roots with the internal `wta-top/kong2022`
producer path; the presentation directory remains named `kong2022_7net`.
The active contrast aggregate is
`contrasts/canonical_all_grids.manifest.json`, whose four grids contain only
`(N,7,400)` subject matrices. The complete six-figure audit is recorded in
`docs/kong400_recovery_audit/visualization_7net_audit.json`.

## Artifacts and verification

- `contrasts/{version}_{grid}_{cohort}_subjects.npz`: float32 `left`, `right`,
  `values` (bilateral), ordered `subjects` and `panels`, plus a provenance
  `.manifest.json`. All four grids occupy about 32.13 MiB compressed.
- `contrasts/{version}_{grid}_{cohort}_vs_iowa_hc.npz` and TSV: all ROI t/p,
  effective N, testable and significant masks (`p<0.05`). HC has no
  self-comparison.
- `contrasts/{version}_all_grids.manifest.json` (Standard) or
  `contrasts/{version}_{grid}*.manifest.json` (explicit branch selection):
  statistics settings, registry, source mode, signal branch, dependencies,
  inputs, outputs and shared t range. All recorded hashes are checked before
  accepting the shared scale.
- Historical recovery artifacts (`contrast_output_qa.json`,
  `contrast_recovery_run.log`, `contrast_input_audit.json`,
  `backup_receipt_20260910T030840Z.json`): independent verification of 44,800
  ROI tests, rendering receipts, and preflight audit archived in git commit
  `bfd84d6`.
- `*.png.backup-20260910T030840Z` and
  `render_multidataset_hipp_cortex_gradient_wta_legacy.py`: preserved
  historical image and legacy script backups retained in-place for reference.
  `.backup-tm-column-20260916T175810Z` siblings preserve the canonical
  two-grid contrast files replaced by the TM-addition rebuild.
- `contrasts/canonical_2mm_36p{,_ctxreg}_tm.manifest.json` and
  `*_tm_subjects.npz(.manifest.json)` / `*_tm_vs_iowa_hc.npz(.tsv)`: TM
  third-column subject FC, Welch contrasts and atlas declarations.

Regression tests are in `tests/visualize/test_cortex_group_contrasts.py`,
`tests/visualize/test_tm_group_contrasts.py` and
the existing multidataset/7net/standalone-CLI suites. They cover Welch direction,
raw p thresholding, missing data, ID alignment, source-mode boundaries, hash
rejection, mean consistency, stale cross-grid results, actual label taxonomy, cohort N,
equal map dimensions and color masking.

# Report renderer

`report.render(results_directory)` reads only eight manifest-declared
statistical derivatives: `statistics.csv`, `subject_metrics.csv`, `curves.csv`,
`boundary_positions.csv`, `spatial_maps.json`, `roi_registry.json`,
`roi_statistics.csv`, and `analysis_config.json`.
`run_manifest.json` must include `statistics` in `completed_stages` and a
matching `outputs[filename].sha256` for every input. Missing files, missing
hashes, changed bytes, duplicate matrix cells, a registry other than exactly
400 distinct ROIs, malformed/duplicate ROI estimates, or a scope mismatch fail
closed. The semantic scope check fixes 2 mm, 36P, canonical7, 400 ROI,
standard/ctxreg, bilateral hippocampi, all three patient comparisons, the
cortex-only shared 36P rows, complete 4-cohort × L/R × atlas descriptive grids,
and every expected ROI effect cell. Individual curves must use the declared
participant cohorts; descriptive atlas curves use only the explicit
`group_hc`, `group_patient`, `group_pure_mdd`, and `group_comorbid_mdd`
identifiers. Re-rendering never reopens scientific source inputs or chooses
alternate artifacts.

Statistic coverage is derived from the hash-validated `subject_metrics.csv`:
every unique `(theme, branch, hemi, atlas, system, metric, target)` base must
have exactly one estimate for each of patient, pure MDD, and comorbid MDD versus
HC. This prevents a partially missing fine-grained matrix from passing a
coarser panel-level scope check.

The renderer writes five theme PNGs, the paginated static
`iowa_hipp_functional_organization_report.pdf`, and `index.html`. Theme strings are
`identity`, `spatial`, `fingerprints`, `continuous`; explicit numeric identifiers
`1`–`4` denote those same themes. All figures use reader-friendly anatomical terms
(Ant/Mid/Post, LH/RH, sample sizes n=26/n=7/n=19/n=67) and include self-contained
one-line metric interpretation summaries in panel headers.

Theme 1 splits Standard and Cortex-Regressed (Ctx-Reg) pipelines into two standalone,
identically laid-out publication figures (`hipp_functional_identity_standard.png` and
`hipp_functional_identity_ctxreg.png`). The dispatcher routes exactly six
(metric, atlas) identity effect panels to this dedicated renderer; any other
panel count raises immediately. The legacy generic-grid fallback for identity
(including its whole-hippocampus subject scatter panels and the block that saved
one shared figure to both pipeline files) has been removed, so a scope mismatch
can never again silently overwrite these two figures. The legacy unversioned `theme_1_identity.png` duplicate
and engineering prefixes (`theme_N_`) have been replaced with self-explanatory scientific content
abbreviations across all themes. Heatmaps and whole-hippocampus subplots
are replaced with six grouped bar + scatter distribution panels for
`smax`, `margin`, `fixed_hcp_similarity`, `fixed_hcp_advantage`, `segregation`,
and `boundary_sharpness`, followed by a full-width dedicated
guide and reference card in the 8th slot. Each metric panel organizes data into major
hemispheric groups (LH vs RH) and longitudinal subregion subgroups (Ant, Mid, Post),
presenting 4 cohort bars (HC n=67, All Patients n=26, Pure MDD n=7, Comorbid MDD n=19)
with mean ± SEM bars and overlaid jittered subject observations. Significant contrasts
versus HC (+, *, **, ***; uncorrected two-sided Welch tests) are annotated with
horizontal brackets and exact Hedges’ g effect sizes (without redundant "g=" text or explicit "+/-" signs, e.g. `* (0.49)`),
color-coded by sign: RED for positive effects (Patient > HC) and BLUE for negative effects (Patient < HC).
Brackets and tick marks are anchored strictly above the local maximum scatter observations of each
subregion with dynamic adaptive headroom to guarantee zero collision with scatter points, while
preserving bar heights across 40–60% of the visible panel span. The full-width reference card features
graphical color swatches, statistical badge explanations, and explicit biological/methodological definitions
for all six functional identity metrics.

Theme 2 (`hipp_spatial_organization.png`) uses a 4-tier balanced publication layout:
- Tier 0: 5 effect matrices (Area Fraction, Physical Surface Area in mm², Longitudinal
  Centroid Position along AP axis, Surface Vertex Count, and Whole-Cortex Network Coverage %).
- Tier 1: 4×4 grid of 16 area fraction profiles along the longitudinal AP axis (0.0 to 1.0)
  with a compact 8-handle proxy legend on panel (0,0) (separating system colors from cohort
  line styles) and duplicate y-axis labels suppressed across columns 1–3 to keep curves clear.
- Tier 2: 1×4 row of individual TM boundary transition midpoint rugs with transition pair
  colors (Ant–Mid, Ant–Post, Mid–Post), black median diamonds, and cohort group dividers,
  suppressing redundant y-tick labels on panels 1–3 to eliminate horizontal text collisions.
- Tier 3: 2×3 grid of 6 descriptive group surface composition maps (Group Dominant, De Novo,
  WTA across Standard and Ctx-Reg) at the 512-vertex geometry with clear cohort rows and left-aligned labels.

Theme 3 (`hipp_network_fingerprints.png`) uses a 2-row × 3-column matrix (Standard vs Ctx-Reg pipelines across
Group Dominant, De Novo, and WTA atlases):
- Heavy structural dividers separate LH and RH (at y=2.5) and Anterior, Middle, and Posterior
  subregions (at x=6.5 and x=13.5).
- Each panel carries a single-line title (`Standard/Ctx-Reg · Group Dominant/De Novo/WTA`);
  pipeline and Fisher's-z metric context lives in the suptitle, and subregion banners
  (Ant/rA in blue, Mid/rB in orange, Post/rP in green) sit directly above their column blocks.
- X-tick network labels reuse the fixed canonical7 hues from `inputs.py`
  (`Default` red for DMN, `Somatomotor` steel-blue for SMN, `VentralAttention`
  magenta for VAN, `DorsalAttention` green for DAN, `Control` orange for CON,
  `Language` brown for LAN, `Visual` purple for VN), darkened only when
  luminance exceeds 0.60 so text stays legible on white; labels are bold.
- Cells are square (`imshow aspect='equal'` + axes `equal`), cell effects use 9.0pt, x-ticks 10pt at 55°,
  y-ticks 10pt, single-line panel titles 13pt, suptitle 15pt; figure size is 36×11.5 with row spacing (hspace 0.45)
  tightened so the middle gap only fits the top-row x-tick labels plus the bottom-row panel titles.
- The bottom horizontal colorbar carries an explicit title naming the plotted quantity
  (`Hedges' g effect size (Patient − HC cortical-network FC, Fisher's z)`) plus a scale/direction label
  and tick labels; arrows still mark clipped colors while exact numeric effects remain.
- A descriptive subtitle annotates the 7 canonical cortical networks (DMN, SMN, VAN, DAN, CON, LAN, VN)
  and their hippocampal subregion alignments. Canvas margins (top=0.86/0.88, bottom=0.12, y=0.975) eliminate suptitle collisions.

Theme 4 (`hipp_continuous_gradients.png`) uses a 4-tier structured layout:
- Tier 0: 5 topological effect matrices (Longitudinal Axis Alignment r, Group Template Concordance r,
  Gradient Dynamic Range G1–G3, Template Dynamic Range, and AP Profile Slope Δr/mm) with self-contained
  metric guides.
- Tier 1: 1×4 row of continuous longitudinal AP profile curves (Standard and Ctx-Reg LH/RH) with
  system color coding (Ant: blue, Mid: orange, Post: green, Post−Ant: purple) and cohort line styles,
  utilizing an 8-handle proxy legend on panel (0,0) without curve obstruction.
- Tiers 2 & 3: Descriptive connectopic gradient maps for Standard and Ctx-Reg pipelines (G1 Principal AP Axis,
  G2 Secondary Subfield, G3 Tertiary) across bilateral hippocampi and 4 cohorts (HC, All Patients, Pure MDD, Comorbid MDD).

All inferential matrices display patient, pure_mdd, and comorbid_mdd against HC,
without significance filtering. Hedges g has a shared zero-centered color range
[-1.5, 1.5]; arrows mark clipped colors while exact numeric effects remain.
Missing values use a gray ×; genuine zeros retain 0.00. Uncorrected p markers
use exact boundaries: + for .05 ≤ p < .1, * for .01 ≤ p < .05, ** for
.001 ≤ p < .01, and *** below .001. Confidence intervals and sample sizes are
available in the HTML hover and exact estimate table.

The static PDF is made only from the four exact newly rendered theme PNGs, in
fixed theme order. It adds a scope and reading-guide cover, then splits tall figures
across standard pages without reopening source inputs or making new inferences. It is
a reading/printing companion; CSV files remain the exact-estimate source.

The HTML is an offline interactive report with embedded data and JavaScript;
its linked PNGs sit alongside it. Branch, hemisphere, atlas, and metric filters
operate on every estimate. The searchable ROI effect table embeds every
manifest-validated ROI result; it initially renders the first 500 matching rows
for responsiveness and has a user-controlled option to render all matches.
The ROI registry exposes every cortical ROI. Declared three-coordinate ROI
centroids, when available under `coordinates`, produce an interactive x/y
localization projection. Otherwise the report explicitly describes localization
by hemisphere and anatomical labels and does not infer a cortical surface.
Network colors come from `config/hipp_network_style.json` with explicit mappings
for DMN, SMN, VAN, DAN, CON, LAN, and VN.

Verification uses the explicit py314 interpreter with single-thread numeric
libraries. `test_report.py` tests exact significance boundaries, missing versus
zero, derivative hash rejection, required ROI statistics, and real generation
of all four PNGs, the static PDF, plus HTML. Test fixtures are created under the system temporary
directory, never inside the source module, so ordinary test runs do not dirty
the worktree.

# Numerical definitions

`metrics.py` returns explicit `NaN` values with reasons for noncomputable
summaries. Missing results are never replaced with zero. The caller supplies an
exactly validated binary vertex mask; invalid vertices do not become an
unassigned label. Areas must be finite and nonnegative; whole-side total area
must be positive. Labels are 0 (unassigned), 1 (A), 2 (B), or 3 (P).

- Individual-TM identity: area-weighted maximum signed similarity and maximum
  minus second-largest similarity. A vertex is eligible only when its validated
  mask is true and all three scores are finite. Whole-side totals retain the
  full side area, while `valid_area` reports the actually eligible area.
  System rows use only valid vertices of the requested individual TM label.
- Fixed-HCP identity: HCP labels and support define one external A/B/P
  territory per branch and hemisphere for every cohort. Within each territory,
  similarity is the matched HCP-template score and advantage is that score
  minus the maximum competing HCP-template score. `total_area` is the whole
  fixed HCP territory; `valid_area` is the evaluable subject area. The HCP
  labels/support must agree exactly; no Iowa group atlas or participant label
  selects the territory.
- Group TM dominant atlas: at every group vertex, majority counts and the
  mean-template-score tie-break include only sources with a true valid mask,
  A/B/P label, and finite three-template scores. A vertex with no usable source
  remains label 0; individual participants are never removed from the roster.
  Descriptive TM-dominant curves/maps use exactly this eligible-source support;
  de novo and WTA descriptive products instead use their own separately
  declared, binary atlas support masks. Unsupported vertices are absent rather
  than encoded as label 0.
- Spatial extent: A/B/P, unassigned area, and vertex count include only valid
  vertices of the corresponding label. Invalid vertices are excluded from all
  four categories rather than reassigned to label 0. Fractions retain the full
  side area as denominator so coverage loss is visible through `valid_area`.
  Centroids use valid, area-weighted finite longitudinal coordinates. Twenty
  fixed bins cover [0,1], left closed/right open except the last bin, which
  includes 1. Spatial curves divide valid system bin area by the full bin area;
  each curve also records valid and total bin area. Empty bins are missing,
  not zero.
- Boundary positions: use every unique native-mesh triangle edge whose two
  endpoints are valid, assigned to different A/B/P labels, and have finite
  unfolded longitudinal coordinates. Each retained row names its unordered
  system pair and records the endpoints' AP midpoint. All rows remain in
  `boundary_positions.csv`; multiple and disconnected transitions are never
  reduced to a unique boundary location.
- Template organization: rA, rB, rP, and rP-rA slopes use weighted least
  squares with vertex area weights and an intercept, over valid finite vertices.
  Robust range is the unweighted finite-vertex 95th minus 5th percentile.
  Curves are area-weighted within the same twenty fixed bins. Every row records
  valid and full-side area.
- FC: form the mean valid regional time series first, correlate that series
  with each of the 400 ROI series, then apply Fisher arctanh. Constant,
  nonfinite, insufficient-length, and exactly perfect correlations are missing;
  there is no arbitrary clipping epsilon. Network FC is the arithmetic mean
  of ROI Fisher-z values in the exact supplied registry, in order DMN, SMN,
  VAN, DAN, CON, LAN, VN. One missing ROI makes its regional-network value
  missing; no partial network average is substituted.
- Segregation: compute each valid vertex's Pearson correlation with a system
  mean time series. Cohen d compares correlations inside that system against
  those outside it, using the sample-size-weighted pooled sample SD. Each
  distribution needs at least two finite values and pooled SD must be positive.
  System means include the evaluated vertex when it belongs to that system.
- Boundary sharpness: consider only surface edges whose two endpoints have
  valid time series and different assigned labels 1–3. A neighborhood is the
  union of vertices within 4 mm of either endpoint, measured by shortest paths
  along triangle edges. This is an **edge-graph geodesic approximation**, not an
  exact continuous-surface geodesic. Traversal follows the supplied anatomical
  mesh; only valid vertices carrying the respective endpoint labels enter the
  two side distributions. Each side must contain at least two vertices; the
  radius never expands. For each side mean time series, compute Cohen d of
  within-side versus opposite-side vertex correlations, then average the two
  directional d values. A system's boundary sharpness is the mean over its
  incident computable boundary edges. `n_valid_boundaries` counts these edges.
- Gradients: fit a five-dimensional orthogonal Procrustes rotation on common
  valid finite vertices without centering, scaling, or gradient normalization.
  Fewer than five vertices or rank-deficient inputs make alignment missing.
  Apply the rotation to valid finite subject vertices. G1–G3 summaries are
  unweighted 95th–5th percentile range, Pearson longitudinal correlation, and
  Pearson correlation with the matching reference gradient. Coordinates may
  be missing; correlation uses paired finite vertices only.
- Group inference: exclude nonfinite participant values independently within
  groups. Use the exact two-sided Welch t test and t-based 95% difference CI.
  Hedges g uses patient minus HC mean, pooled sample SD, and the exact gamma
  correction J(n_patient+n_HC-2). Groups with fewer than two observations or
  zero standard error receive an explicit reason. Significance thresholds are
  `+` for .05 ≤ p < .1, `*` for .01 ≤ p < .05, `**` for .001 ≤ p < .01,
  `***` for p < .001; p ≥ .1 is blank and missing p is `NA`.
- Cortical area background: only explicit canonical7 `component_kind=network`
  rows enter the bilateral cortical summary. The selected participant IDs must
  equal the declared HC, Patient, Pure MDD, and Comorbid MDD rosters exactly;
  matching only the expected count is insufficient.

Run numerical checks with one numerical-library thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m unittest discover \
  -s scripts/visualize/iowa_hipp_functional_organization -p test_metrics.py
```

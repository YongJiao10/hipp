# Shifted-Cosine Cross-Fit Design

Gradient-spectral group and cross-fit clustering use one shared surface-edge
affinity implementation. As of the formal 2026-07-13 promotion, the only
supported definition is shifted cosine, `(1 + cosine_similarity) / 2`,
multiplied by the hippocampal surface adjacency. The experimental compatibility
switch described by the original design has been retired.

The HCP experiment was initially isolated from official outputs. It first builds all five
family-aware templates and held-out subject labels, then evaluates a fixed gate:
all affinity graphs connected, every cluster at least 10% of support, pairwise
macro-Dice at least 0.85, ARI at least 0.70, and prototype correlation at least
0.50. Group comparisons run only after this gate passes.

The comparison stage reuses saved gradients and FC inputs for HCP, Stanford HC,
and Stanford Symp. It rebuilds only the default RouteA G1-G3, K=3, coverage 0.80
group products and fisherz/count overviews. The accepted shifted-cosine
cross-fit is promoted to canonical outputs only after pre-formalization
gradient-derived products are archived.

After visual review, derive weighted-fisherz products from the accepted fixed
K=3 shifted-cosine group roots for HCP 151, Stanford HC, and Stanford Symp.
Use the maintained `05_05` weighted branch builder and `06_01` renderer with
dataset-explicit roots. This stage reuses labels and subject surface time series
and does not refit FC gradients or K selection.

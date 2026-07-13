# Shifted-Cosine Cross-Fit Design

Gradient-spectral group and cross-fit clustering use one shared surface-edge
affinity implementation. The maintained default is shifted cosine,
`(1 + cosine_similarity) / 2`, multiplied by the hippocampal surface adjacency.
The previous ReLU-cosine definition remains available explicitly as
`relu-cosine` for historical reproduction.

The HCP experiment is isolated from official outputs. It first builds all five
family-aware templates and held-out subject labels, then evaluates a fixed gate:
all affinity graphs connected, every cluster at least 10% of support, pairwise
macro-Dice at least 0.85, ARI at least 0.70, and prototype correlation at least
0.50. Group comparisons run only after this gate passes.

The comparison stage reuses saved gradients and FC inputs for HCP, Stanford HC,
and Stanford Symp. It rebuilds only the default RouteA G1-G3, K=3, coverage 0.80
group products and fisherz/count overviews. Existing results remain read-only;
replacement and archival are separate decisions.

After visual review, derive weighted-fisherz products from the accepted fixed
K=3 shifted-cosine group roots for HCP 151, Stanford HC, and Stanford Symp.
Use the maintained `05_05` weighted branch builder and `06_01` renderer with
dataset-explicit roots. This stage reuses labels and subject surface time series
and does not refit FC gradients or K selection.

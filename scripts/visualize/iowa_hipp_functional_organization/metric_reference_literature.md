# Reference choice for hippocampal functional organization metrics

Literature assessment, 2026-09-18. Primary-source HTML was consulted; no PDFs or archive content were accessed. This note assesses design principles, not the implementation or statistical significance of the current results.

## Recommended interpretation

A fixed external HCP connectivity template and fixed spatial territory provide a common reference for testing canonical organization across groups. A group-specific reference instead measures resemblance to that group's organization. Neither is intrinsically meaningless, but they answer different questions. Re-centering each group around its own organization can obscure a systematic displacement from the common reference. This is methodological reasoning, not a published result about this cohort.

Keep three objects distinct: (1) the cortical connectivity vector used as a functional template, (2) a hippocampal atlas supplying the territory over which a metric is summarized, and (3) subject-specific labels estimated by maximizing template similarity. Changing only the territory does not change the connectivity template. A shared template combined with group-specific territories can still produce incomparable spatial summaries.

For a primary common-reference endpoint, freeze both the HCP connectivity templates and the HCP territories before group comparisons. Summarize similarity to the designated template within its fixed territory. Complement this with designated-minus-competing-template similarity and spatial displacement/overlap. Recomputing a maximum after reassignment can preserve high fit despite a relocated functional system. Independent-session evaluation is preferable when a subject's data determine both labels and the reported fit; an external template alone does not remove that within-subject selection dependence.

An HCP mean template is an external reference, not a normative model. Lower similarity establishes lower conformity to that reference; it does not by itself establish disorganization, dysfunction, disease specificity, or a causal mechanism. Healthy individual variation, age, data quality, acquisition differences, registration, and clinical heterogeneity remain alternatives. HC > patient similarity is a hypothesis to test, not an expected result required for publication.

## Primary-source findings

- **Kember et al., 2026:** Their precision-mapping method derives cortical connectivity templates and assigns hippocampal vertices by maximum template correlation. Segregation is Cohen's d comparing correlations with a regional mean BOLD signal inside versus outside the region; boundary sharpness localizes this comparison around border vertices within 4 mm. An independent HBN cohort reused HCP-D templates. Topographic developmental effects replicated, but segregation and boundary-sharpness age effects did not. This supports separating topography from functional differentiation and scrutinizing acquisition sensitivity. It does not establish the expected direction in adult MDD. [Nature Communications, original article](https://www.nature.com/articles/s41467-026-74572-1).

- **Kriegeskorte et al., 2009:** Selection and subsequent testing on the same noisy observations can bias descriptive results and invalidate inference when they are not independent under the null. In the present proposed design, label selection by maximum correlation and reporting that selected correlation deserve a specific independence check. Likewise, including a person in their own group template can inflate apparent resemblance; exclusion/cross-validation is needed if the estimand is out-of-sample fit. [Original methodological article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/).

- **Marquand et al., 2016:** Normative modeling estimates expected biological variation conditional on relevant predictors and provides predictive uncertainty, including for excluded or new participants. A single template-correlation value lacks this calibrated reference distribution. Clinical deviations also need not align uniformly across patients. [Biological Psychiatry, original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5023321/).

- **Chan et al., 2014:** Functional system segregation varied with healthy adult aging and related to memory. Thus segregation has interpretable systems-level meaning, but its normal variation and behavioral context matter. Their cortical segregation metric must not be assumed identical to Kember's hippocampal Cohen's d metric. [PNAS, original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4246293/).

- **Przezdzik et al., 2019:** Gradual longitudinal hippocampal connectivity organization captured individual variation related to recollection. Fixed parcels can summarize canonical territories but should not be interpreted as proving that the underlying biology is sharply discrete. [Cortex, original study](https://www.sciencedirect.com/science/article/pii/S0010945219301832).

- **Ciric et al., 2017:** Confound-regression approaches differ in residual motion artifact, distance dependence, and degrees-of-freedom costs. Group differences in template fit or segregation require inspection of residual motion and usable data, not just an identical preprocessing label. [NeuroImage, original benchmarking study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5483393/).

## Practical inference limits for this figure

The displayed All Patients group includes the Pure and Comorbid MDD groups, so these are not four independent samples. Pure MDD n=7 supports only imprecise subgroup estimates; report intervals and avoid treating nonsignificance as equivalence. Do not pick a reference or endpoint because it produces the preferred ordering. Freeze a small endpoint family, retain two-sided uncertainty and multiplicity control, and use alternative reference definitions as clearly labeled sensitivity analyses. Distinguish reduced canonical conformity, shifted topography, and reduced within-versus-between differentiation: they can coexist or dissociate and should not be collapsed into a single claim of functional disorganization.

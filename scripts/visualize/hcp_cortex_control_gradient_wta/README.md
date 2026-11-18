# HCP 7T Local Cortex Regression Comparison

Compares hippocampal gradient and winner-take-all (WTA) organization between the **Standard** branch and the **CtxReg** (local cortex regression) controls at **10 mm** and **20 mm** in HCP 7T young adults, for 2mm and 4mm hippocampal smoothing in one figure. The cohort size is the length of the HCP QC-pass roster and is recorded in the group-atlas manifests.

---

## 1. Branches

1. **Standard**
   - Group atlas root: `outputs_hcp/work/group_atlas/pfm`
   - Uncontrolled hippocampal time series and the Kong2022 400-ROI FC bank.
2. **CtxReg 10 mm**
   - Group atlas root: `outputs_hcp/work/group_atlas/pfm_ctxreg_10mm`
   - For each hippocampal vertex and run, the mean of valid cortex within 10 mm (native T1w distance) is regressed out of the **hippocampal** time series (`04_00 --route-tag pfm_ctxreg_10mm`); cortical ROI references are unchanged.
3. **CtxReg 20 mm**
   - Group atlas root: `outputs_hcp/work/group_atlas/pfm_ctxreg`
   - Same regression with a 20 mm radius (`04_00 --route-tag pfm_ctxreg`).

The former CtxDrop branch (excluding local cortex from the ROI targets) is retired. Its effect is quantified at the FC/WTA level by `scripts/core_workflow/06_07_render_local_cortex_control_diagnostics.py`, which shows that ROI-level targets are essentially unaffected by local exclusion and that CtxReg changes are driven by the signal shared within about 3 mm of the hippocampus.

---

## 2. Layout and modalities

The top block shows 2mm and the bottom block 4mm hippocampal smoothing; each block has one row per branch (Standard, CtxReg 10 mm, CtxReg 20 mm) and one column per modality (Fisher-z, OvR, normalized top-20, WTA).

Gradient panels read the formal HCP gradient run shared with the multi-dataset renderers (`HCP_KONG_GRADIENT_RUN` in `scripts/visualize/lib/multidataset_cohorts.py`: the TopSim profile and its primary Ward partition, `group_atlas/<branch>/topsim/gradient-ward/kong2022/<smoothing>/cov_0p80/g1g2g3/`). All three gradient columns use the same Ward K = 3 clusters and differ only in the network annotation. WTA panels read the clusterless `group_atlas/<branch>/wta-top/kong2022/<smoothing>/cov_0p80/`.

1. **Gradient clustering — positive Fisher-z FC profiles**: 4-view Workbench composite of the k=3 clusters labelled by dominant positive Fisher-z network, with left/right stacked bars of positive Fisher-z per canonical7 network for C1 (anterior), C2 (middle) and C3 (posterior). The Fisher-z axis is scaled per branch.
2. **Gradient clustering — OvR logistic model (network time-series features)**: the same clusters labelled by the descriptive one-vs-rest logistic model `ovr_logistic_canonical7_network_ts`, whose features are subject-by-cluster Fisher-z FC with each canonical7 network time series. Stacked bars show the positive raw coefficients per network; the label is the largest strictly positive raw coefficient (`cluster_assignment_status_ovr_logistic_canonical7_network_ts.json`). The coefficient axis is scaled per branch. The ROI-feature OvR (`ovr_logistic_canonical7`) agrees closely (coefficient profiles median r = 0.99 across branches, smoothings, hemispheres and clusters; 35 of 36 labels identical, the exception being CtxReg 10 mm, 2mm, right C1: DMN vs SMN with near-tied coefficients) and is reported in the group overviews.
3. **Gradient clustering — normalized top-20 network recruitment (%)**: clusters labelled by normalized top-20 count, with stacked bars of the normalized top-20 composition per subregion (counts divided by each network's available ROIs).
4. **WTA parcellation — 20 long-axis bins (%)**: canonical7 vertex assignments and the network composition of 20 anterior-to-posterior bins.

Network abbreviations: DMN default mode, SMN somatomotor, VAN ventral attention, DAN dorsal attention, CON control, LAN language, VN visual (Kong2022 canonical7).

---

## 3. Outputs

- `hcp_cortex_control_gradient_wta.png`: the combined 2mm/4mm figure (about 64 in wide for four columns; 2.25 in header, 4.8 in per row for six rows, 1.35 in legend zone).
- `stats/hcp_cortex_control_gradient_wta_stats.tsv`: plotted values with `branch` and `smoothing` columns; modalities `gradient_fisherz`, `gradient_ovr_network_ts` (positive raw coefficients; `margin` holds the winning coefficient), `gradient_top20` and `wta_longaxis`.

---

## 4. Execution

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py
```

Options: `--force` (re-render cached panels), `--out-png <path>`, `--stats-tsv <path>`.

---

## 5. Panel cache provenance

Surface panels are cached under `panels/` with SHA-256 `.sources.json` sidecars recording the template surfaces, label GIFTIs and their contents, the spec file, the locked view scene (`config/wb_locked_native_view_lateral_medial.scene`) and the output PNG hash. A change in any upstream label or manifest invalidates the cache; `--force` re-renders unconditionally.

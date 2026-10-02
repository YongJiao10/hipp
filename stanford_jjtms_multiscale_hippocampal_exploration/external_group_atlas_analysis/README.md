# Stanford group atlas descriptive comparison

> **KEMBER_REPLICATION_STATUS=group_geometry_only**
>
> 本工作流只描述 group-specific K=3 atlas 的几何和标签一致性，不产生 Kember
> 个体级面积、functional segregation 或 boundary sharpness 结果。
> `atlas_region_area_*` 是 atlas geometry namespace，不能与
> `kember_metrics` 的 native HippUnfold `surfareacorobl` 面积互换。


This is the group-atlas geometry family used by the unified
`cross_dataset_rest_fc` run. The current `routea` K=3 registry contains the
explicit `HC` reference plus seven configured Stanford cohort atlases:

- parent factorial groups: `NTHC`, `TEHC`, `NTS`, `TIS`;
- nested clinical cohorts: `NTS Any MDD`, `NTS Pure MDD`, `TIS PTSD`.

The four parent groups remain the formal trauma-by-symptom factorial design.
The nested cohorts are overlapping clinical subsets, not additional mutually
exclusive factorial cells. All eight maps are valid inputs for this descriptive
atlas family.

It reads exactly

`8 groups (HC plus seven cohorts) × 2 assignment variants (plain/weighted) × 2 smoothings (2/4 mm) × 2 hemispheres = 64 atlas cells`

from `outputs_stanford/work/group_atlas/`. Each input must pass the manifest
gate (`status=ok`, `chosen_k_final=3`, ordered labels `[1,2,3]`) and use the
manifest's `template_left_or_right_surface` as the consumed 512-vertex, 930-face
midthickness geometry. Density evidence is copied into the inventory and table
outputs rather than inferred from filename arguments.

The unified primary run reports:

- `atlas_inventory.csv`: validated 64-cell inventory and provenance;
- `region_area.csv`: cluster 1/2/3 `atlas_region_area_mm2`,
  `atlas_region_area_fraction`, and `atlas_total_surface_area_mm2` using the
  explicit atlas triangle-area/3 geometry convention;
- `matched_group_pair_comparison.csv`: seven non-HC-versus-HC pairs within each
  matched specification, with raw label agreement, ARI, ordered cluster Dice,
  boundary Dice, and symmetric mesh-edge geodesic mean/95th/Hausdorff distances
  for 1–2 and 2–3 boundaries;
- `within_group_spec_stability.csv`: 2-vs-4 mm and plain-vs-weighted
  specification comparisons as descriptive noise references;
- `descriptive_contrasts.csv`: seven direct non-HC-minus-HC summaries for area
  fractions, boundary position, and boundary distance.

Comparison tables retain both consumed geometry paths. Matched cohort cells use
the same geometry. Plain-vs-weighted rows have distinct HCP/Stanford consumed
surfaces: vertex-index label agreement/ARI/Dice remain descriptive, but physical
boundary distance and position are NA rather than computed on map A geometry.

The authoritative run is `cross_dataset_rest_fc/*/primary_screen/`; this
directory remains the implementation and compatibility CLI location for the
geometry family.

When a correction is found, rerun the same current target; its tables, figures, and report are overwritten together so the directory always describes one current analysis.

Run with the required py314 interpreter and the current run id:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/run_atlas_analysis.py \
  --run-id atlas_descriptive
```

This compatibility command writes diagnostics under this directory's `results/<run_id>/`,
`figures/<run_id>/`, and `reports/<run_id>/`; it is not the experimental
scheduler or an independent authoritative run. The unified `primary_screen`
writes the authoritative atlas tables under `cross_dataset_rest_fc/results/primary_screen/group_atlas/`.
Reruns overwrite the current compatibility files in these directories. This is a promising descriptive screen only: it has no
uncertainty, p/q values, split-half/ICC reliability gate, or direct group-level
atlas inference. It does not modify the primary FC result or access beta, CON, native, or archive data.

## 分支边界

本目录的 region-area、标签一致性和边界表都是 group-level 几何描述。HC 加七个
cohort map 可用于 atlas 描述，但不回填 `kember_metrics` 的 subject-level
feature table；它们由 `cross_dataset_rest_fc` 的共享 branch ledger 调度，
但不改变 FC 的四组 factorial 数值推断。

Group-specific clustering 是统一运行中的 geometry family。它不进行个体聚类，
group-wise 只保留每个非 HC 组对 HC，nested cohorts 也不被当作新的互斥
factorial cells。

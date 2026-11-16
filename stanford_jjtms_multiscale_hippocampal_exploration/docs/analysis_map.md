# Stanford/JJTMS analysis map

The following is the frozen boundary between the active analysis families. The
subject-level FC families and the group-atlas geometry family share one branch
registry, strict priority barrier, run manifest, and output closure. Their
rows retain distinct analysis-family metadata and are not silently reused as
inputs to another estimand.

| Workflow | Source code | Inputs | Estimand | Current outputs | Forbidden cross-usage |
| --- | --- | --- | --- | --- | --- |
| `cross_dataset_rest_fc` | cross_dataset_rest_fc/ | subject-level rest-FC, shared labels, cohort/design manifests | network × hippocampal-system Fisher-z FC differences and Stanford four-parent-group factorial contrasts | primary FC localisation, profile summaries, contrast tables and technical reports | Do not use FC results as area, boundary, or atlas geometry endpoints; do not replace shared labels with group-trained labels for the primary FC comparison |
| `kember_metrics` | kember_metrics/src/ | 512-source rest-fMRI/labels plus native HippUnfold surfareacorobl, aligned native labels, validity mask and provenance | subject-level Kember area, functional segregation and boundary sharpness for fixed and template-matching methods | one combined feature table, one combined direct-contrast table, figures and manifests at stable package paths | Do not use geometry-derived triangle-area/3 reconstruction or transition_area as Kember area; do not use group-level atlas areas as subject metrics |
| `external_group_atlas_analysis` | external_group_atlas_analysis/ | HC plus seven parent/nested group-specific atlas labels, consumed surfaces and manifests | group_atlas_gradient_spectral group-level geometric description of atlas region areas, label agreement, and boundary geometry | 64-cell inventory, atlas_region_area_mm2, seven non-HC-versus-HC similarity/agreement summaries per specification, and descriptive reports under the unified primary run | Do not use atlas geometry rows as subject-level FC values or Kember inference; do not create non-HC all-pairs comparisons |

The two active Kember methods are hcp_template_matching and hcp_fixed; no third subject-level method is defined.
六个 contrast 在两种 method、两种 smoothing、两侧、三类 metric 和三个 system 上形成 432 个 method-aware endpoint rows。

The atlas registry contains the explicit `HC` reference plus the four parent factorial groups `NTHC`, `TEHC`,
`NTS`, and `TIS`, plus the nested clinical cohorts `NTS Any MDD`,
`NTS Pure MDD`, and `TIS PTSD`. The nested cohorts are overlapping clinical
subsets, not additional mutually exclusive factorial cells; all eight maps are
valid descriptive atlas inputs. Group-wise atlas output contains only each
non-HC group versus HC.
Kember workflow 明确**禁止个体聚类**；group-specific clustering 是统一 Rest-FC
调度中的 geometry family，但不是 Kember 的第三个 subject-level method。
The group-atlas atlas_region_area_mm2 is not interchangeable with Kember
surface_area_mm2: the former describes a group-level atlas geometry, while the
latter is a subject-level sum of native HippUnfold per-vertex metric values.

## Unified execution priority

| Tier | Subject-level FC branches | Group-atlas geometry branches |
| --- | --- | --- |
| P0 | 8-network × 2 mm | 2 mm |
| P1 | 8-network × 4 mm and 17-network × 2 mm | none |
| P2 | 17-network × 4 mm | 4 mm |

The barrier is strict: every branch in a higher tier must finish before any
lower-tier branch starts. A failed high-priority branch stops the run and
leaves lower tiers blocked. The retired individual-gradient branch is absent
from the registry.

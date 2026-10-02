# Frozen Kember area method

## Decision

The single area source for all later code is `area_source = hippunfold_native_surfareacorobl`.
The canonical endpoint is the sum of native HippUnfold per-vertex surface-area values
within each native system label. Absolute area in mm² is the Kember-aligned primary
endpoint; a fraction is secondary context.

## Evidence-based source choice

The decision is anchored to the [Kember et al. Nature Communications article](https://www.nature.com/articles/s41467-026-74572-1),
the author [precision-mapping release](https://github.com/JonahKember/precision_mapping)
and its [Zenodo source data](https://doi.org/10.5281/zenodo.20086215), and the
[HippUnfold output documentation](https://hippunfold.khanlab.ca/en/latest/outputs/output_files.html).
Kember defines system area by summing the HippUnfold surface-area value for the
vertices assigned to a functional system. The native metric is selected because it
is the metric consumed by the Kember-aligned workflow and preserves the upstream
per-vertex area convention.

| Candidate | Input | Operation | Status |
| --- | --- | --- | --- |
| retired geometry reconstruction | consumed midthickness coordinates/faces | triangle area / 3 per vertex, then sum by label | geometric QC concept, not Kember |
| kember_metrics/src/surface_area.py | native HippUnfold `surfareacorobl` and native labels | sum native mm² values by system | canonical Kember implementation |

`compute_vertex_area` describes a retired geometric-only comparison and is not
part of the active package. The active `surface_geometry.py` only loads consumed
coordinates/faces and builds adjacency; it cannot silently supply the Kember
endpoint. `transition_area` is not a Kember endpoint; any transition or boundary
quantity is a separately named exploratory geometric/functional measure.

The semantic guardrails are part of the contract:

| Term | Frozen meaning |
| --- | --- |
| `transition_area` | not a Kember endpoint; any use is a separately named exploratory measure |
| `compute_vertex_area` | retired geometric-only comparison; it cannot supply the canonical Kember area |

## Field contract

Every subject, hemisphere, and system record must define these exact fields:

```text
surface_area_mm2
surface_area_fraction
total_surface_area_mm2
total_valid_surface_area_mm2
total_labeled_area_mm2
unassigned_surface_area_mm2
vertex_area_source
vertex_area_metric_path
vertex_area_convention
```

The required formula is:

```text
surface_area_fraction = surface_area_mm2 / total_valid_surface_area_mm2
```

The denominator includes all valid native ROI vertices, including valid vertices
without a system label; invalid/out-of-ROI vertices are excluded. The absolute
surface area is the Kember-aligned primary endpoint. The fraction is secondary
context and must not replace the absolute endpoint.

| Field | Definition |
| --- | --- |
| `surface_area_mm2` | Sum of native per-vertex area values for one system label in one subject and hemisphere; this is the absolute Kember-aligned primary endpoint. |
| `surface_area_fraction` | `surface_area_mm2 / total_valid_surface_area_mm2`; a secondary context field whose denominator includes valid unlabeled ROI vertices. |
| `total_surface_area_mm2` | sum of the consumed native per-vertex metric values over every consumed vertex before validity or label filtering. |
| `total_valid_surface_area_mm2` | sum of consumed native per-vertex metric values at valid ROI vertices; this is the fraction denominator. |
| `total_labeled_area_mm2` | sum of the three system surface areas for labels 1, 2, and 3 at valid native ROI vertices. |
| `unassigned_surface_area_mm2` | `total_surface_area_mm2 - total_labeled_area_mm2`; this includes valid label-0 and invalid/out-of-ROI area. |
| `vertex_area_source` | Identifier stating that the area values came from the native HippUnfold per-vertex metric, rather than mesh-derived geometry. |
| `vertex_area_metric_path` | Path to the exact consumed native `surfareacorobl` metric asset. |
| `vertex_area_convention` | Native per-vertex surface-area values in mm², summed over vertices selected by the native system label. |

The totals are intentionally related but not interchangeable: labeled area is a
subset of valid area, valid area is a subset of the consumed metric area, and the
unassigned field is the full consumed-area residual after labeled area is removed.

| Native provenance bundle | Every Kember output record and its manifest must have the native area metric, consumed native surface, native labels, label-transfer method, density evidence, and input hashes recorded together. |

This native provenance bundle is mandatory alongside the numeric fields, so a
result cannot be interpreted without the surface, label, transfer, density, and
input identity that produced it.

`vertex_area_source` records that the values came from the HippUnfold native
metric; `vertex_area_metric_path` records the consumed asset; and
`vertex_area_convention` records native per-vertex mm² values summed over the
vertices selected by the native label. A mesh-derived value may be recorded only
as a QC comparison, never as the canonical source.

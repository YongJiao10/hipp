# Kember source audit

Status: `KEMBER_METHOD_NATIVE_AREA_WORKFLOW_READY`

## Frozen source scope

The upstream source checkout is recorded in [source_lock.json](source_lock.json). Its
tracked notebook reads precomputed `ant_mm`, `body_mm`, and `post_mm` fields and
uses them for figures. It does not contain a runnable regional-area generator,
raw HippUnfold surfaces, or regional labels. The bilateral convention is clear:
the combined area is the arithmetic mean of left and right values.

## Metric roles and normalization

The Kember-aligned primary endpoint is absolute system surface area. For each
subject and hemisphere, the workflow sums the consumed native HippUnfold
`surfareacorobl` value over vertices assigned to a system, retaining units of
mm^2. Native-space measurement preserves individual differences in total
hippocampal size; it does not imply normalization by that total.

Relative system area is retained as a secondary endpoint:

```
system_area_fraction = system_area_mm2 / total_valid_surface_area_mm2
```

`total_valid_surface_area_mm2` is the sum of all native per-vertex areas inside
the valid native hippocampal ROI. A valid vertex with label 0 remains in this
denominator; vertices outside the valid ROI do not. The full consumed metric
area (`total_surface_area_mm2`), valid-ROI area, labelled system area, and
unassigned area are recorded separately.

Formal individual-difference analyses therefore retain both absolute and
relative area: absolute area aligns with the Kember surface-area endpoint,
while relative area helps distinguish subregion reallocation from differences
in total hippocampal size. Bilateral relative area is defined locally as the
arithmetic mean of the left and right hemisphere fractions.

## Local Kember-method adaptation

The current workflow therefore makes every input and adaptation explicit:

- use `hcp_template_matching` first, `hcp_fixed` second, and a shared
  group-specific clustering solution third;
- require actual native HippUnfold `surfareacorobl` per-vertex area assets;
- transfer each fixed 512 label field to native vertices with Workbench
  `ADAP_BARY_AREA` in HippUnfold unfolded midthickness space;
- sum the native mm² values within labels 1/2/3; and
- average the two hemisphere values for each system.

The area-weighted surfaces used by `ADAP_BARY_AREA` are only part of categorical
label transfer. The area endpoint itself is the sum of the consumed native
HippUnfold metric values.

## Guardrails

- No per-subject clustering is permitted.
- A missing native area metric or missing native/512 alignment surface excludes
  a subject from the frozen common cohort.
- This workflow reports system surface area; it does not define a boundary-area
  endpoint.
- Each run records input hashes, density evidence, label-transfer provenance,
  and output hashes in the stable current result directory; reruns replace its files.

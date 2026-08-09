# JJ_TMS stimulation-site mapping to individualized Kong2022 networks

## Outcome

The maintained analysis maps each JJ_TMS native-volume stimulation target to
the subject's individualized Kong2022 cortical parcellation. The output keeps
all 70 subjects in the fixed Kong2022 roster:

- 68 subjects have a valid native coordinate and a successful surface/network
  assignment.
- `sub-3003` has no matching `record_subject_id` in
  `TMSfMRI_DataSharing.xlsx`.
- `sub-3026` has a workbook record but no `native_x/native_y/native_z`
  values.

The two unmapped subjects remain explicit rows. Their network fields are not
imputed.

## Volume-to-surface coordinate contract

The individualized labels are 32k fsLR topology, but the corresponding
subject midthickness GIFTI stores vertex coordinates in that subject's native
T1w world geometry. Therefore the mapping does not require converting the
parcellation to a volume:

1. Read `native_x/native_y/native_z` from the workbook.
2. Interpret the triplet as T1w world coordinates in RAS+ millimetres.
3. Select the left or right surface from `visit2_target` and require its
   hemisphere to agree with the sign of `native_x`.
4. Find the closest vertex on
   `sub-<id>_hemi-<L|R>_space-fsLR_den-32k_midthickness.surf.gii`.
5. Read the parcel ID at the same vertex in `lh_labels.txt` or
   `rh_labels.txt`.
6. Join that parcel ID to the canonical Kong2022 registry to report both the
   17-network family and merged 8-network family.

The native coordinates are not treated as voxel indices. If a future workbook
contains IJK indices, those indices must first be multiplied by the matching
native T1w NIfTI affine to obtain RAS+ millimetres. The current result supports
the RAS-mm interpretation: among the 68 mapped subjects, nearest-surface
distance ranges from 0.293 to 4.298 mm, with median 1.480 mm and no value above
5 mm. Every consumed pointset must also be marked as anatomical midthickness
for the expected hemisphere, contain exactly 32,492 vertices per hemisphere,
and carry an identity GIFTI coordinate transform. A density/cardinality or
transform mismatch is rejected rather than silently ignored.

The workbook reader also requires the complete coordinate/provenance schema,
including on an empty worksheet; a missing header is a hard error rather than
being interpreted as a cohort with no matching records.

The workbook MNI coordinates are preserved in the CSV as provenance but are
not used for the primary assignment. Mapping an MNI coordinate directly to a
native T1w surface would mix spaces. If native coordinates are unavailable,
a separately validated MNI-to-T1w point transform is required; this workflow
does not silently substitute one.

## Primary assignment and boundary QC

The primary network is the network of the closest 32k midthickness vertex:

- `parcel_id` and `parcel_name`: individualized Kong2022 parcel.
- `family17`: Kong2022 17-network family.
- `canonical8`: repository-standard merged 8-network family.

A 5-mm Euclidean neighborhood around the target is reported as sensitivity
QC. `neighborhood_primary_family17_fraction` is the fraction of valid
vertices in that neighborhood that share the primary 17-network family.
`neighborhood_dominant_family17` reports the local majority family.

The primary assignment is not replaced by the local majority. A low primary
fraction or disagreement means that the site is close to an individualized
network boundary and should be treated as boundary-sensitive. In the current
analysis, 12 subjects have primary-family fraction below 0.5, and 13 have a
neighborhood majority that differs from the closest-vertex family. The exact
subject lists are stored in the JSON summary.

## Common-atlas comparison without reprojection

The individualized result CSV already records each target's selected
hemisphere and full-surface fsLR-32k `vertex_index`. The common-atlas
comparison reuses that saved index directly; it does not reread subject
surfaces and does not repeat the native-coordinate nearest-vertex search.
At the identical topology vertex, it reads the group/common Kong2022 dlabel
and reports:

- `group_parcel_id`, `group_parcel_name`
- `group_family17`, `group_canonical8`
- `individual_vs_group_same_parcel`
- `individual_vs_group_same_family17`
- `individual_vs_group_same_canonical8`

The comparison preserves the two unmapped roster rows without imputing group
labels. For the 68 mapped subjects, the common atlas agrees with the
individualized assignment in 44/68 cases (64.7%) at `family17` level and
47/68 cases (69.1%) at `canonical8` level. Exact disagreement subject lists
and individual-to-group transition counts are stored in the comparison JSON.

Run the comparison only after the individualized result exists:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/map_jj_tms_sites_to_kong2022.py \
  --compare-existing-individual-results
```

This creates separate artifacts and leaves the individualized outputs intact:

- `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_individual_vs_group_atlas.csv`
- `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_individual_vs_group_atlas_summary.json`

## Inputs

- Workbook:
  `outputs_iowa_jj_tms/TMSfMRI_DataSharing.xlsx`
- Fixed roster:
  `manifests/jj_tms_kong2022_70_subjects.txt`
- Subject surfaces:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf/sub-*/anat/`
- Individualized labels:
  `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022/sub-*/`
- Canonical parcel/network registry:
  `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`
  and `config/cross_atlas_network_merge.json`

## Command

Run from the repository root with the required core-workflow interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/map_jj_tms_sites_to_kong2022.py
```

The writer refuses to overwrite an existing CSV or JSON. Move or rename a
previous result intentionally before rerunning; do not delete outputs without
the required explicit deletion approval.

## Outputs

- `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_tms_site_networks.csv`
- `outputs_iowa_jj_tms/present/tms_site_networks/jj_tms_kong2022_tms_site_networks_summary.json`

The JSON summary records the resolved workbook, roster, surface root, label
root, canonical Kong2022 dlabel, and network merge-config paths. The CSV has
one row per roster subject. Important fields are:

| Field | Meaning |
|---|---|
| `mapping_status` | `mapped`, `missing_workbook_record`, or `missing_native_coordinates` |
| `target_to_surface_distance_mm` | Euclidean distance to the selected midthickness vertex |
| `parcel_id`, `parcel_name` | Individualized Kong2022 parcel at the closest vertex |
| `family17` | Primary Kong2022 17-network family |
| `canonical8` | Primary merged 8-network family |
| `neighborhood_primary_family17_fraction` | Local support for the primary family within 5 mm |
| `neighborhood_dominant_family17` | Local 5-mm majority family |
| `qc_distance_gt_5mm` | Geometry warning |
| `qc_neighborhood_primary_fraction_lt_0p5` | Network-boundary warning |
| `qc_neighborhood_dominant_disagrees` | Closest vertex and local majority differ |

For the 68 mapped subjects, the primary merged-network counts are Default 28,
Control 21, VentralAttention 16, and Language 3.

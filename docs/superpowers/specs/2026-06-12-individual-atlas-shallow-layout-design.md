# Individual Atlas Shallow Layout Design

## Goal

Replace the method-first `method/subjects/sub-*` output tree with a subject-first
layout that is faster to browse while preserving all individual gradient and WTA
artifacts.

## Layout

```text
individual_atlas/
  sub-<id>/
    gradient/
      <2mm|4mm>/<L|R>/
    wta/
      <2mm|4mm>/<L|R>/
  _runs/
```

Each terminal directory keeps the existing files unchanged. Run logs, summaries,
heartbeats, and handoff files move from `gradient_spectral_k3/_runs` to the
top-level `_runs` directory.

## Code Contract

`gradient_combo_output_dir()` and `wta_combo_output_dir()` are the single source
of truth for computation and overview input paths. `manifest_rows()` scans the
new subject-first gradient tree. Resume behavior continues to require successful
gradient and WTA manifests for the same subject/smoothing/hemisphere combination.

## Migration And Safety

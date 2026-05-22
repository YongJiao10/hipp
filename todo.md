# TODO

## Flatten `outputs_core/inputs/hcp7t_hippocampus`

`outputs_core/inputs/hcp7t_hippocampus` currently acts mostly as a redundant namespace layer.

Current major children:

- `outputs_core/inputs/hcp7t_hippocampus/hippunfold`
- `outputs_core/inputs/hcp7t_hippocampus/kong2022`
- `outputs_core/inputs/hcp7t_hippocampus/kong2019_fast_pfm_t1only`
- `outputs_core/inputs/hcp7t_hippocampus/qc`

Potential target layout:

```text
outputs_core/inputs/
  hippunfold/
  kong2022/
  kong2019_fast_pfm_t1only/
  qc/
```

Goal: truly remove the redundant `hcp7t_hippocampus` level from the active
directory structure. This is not meant to be solved by permanent short-path
symlinks.

Do not do a blind `mv` without a migration pass. The old `hcp7t_hippocampus`
path is embedded in scripts, docs, tests, and many generated manifest files
(`stage_manifest.json`, `reference_summary.json`, etc.), including absolute
paths.

Required later migration:

1. Move the active directories up one level:
   `outputs_core/inputs/hcp7t_hippocampus/{hippunfold,kong2022,kong2019_fast_pfm_t1only,qc}`
   -> `outputs_core/inputs/{hippunfold,kong2022,kong2019_fast_pfm_t1only,qc}`.
2. Update script defaults, tests, launchers, docs, and environment/reproducibility
   manifests to use the new paths.
3. Update generated manifests or explicitly archive them as historical
   provenance.
4. Validate key workflows against the new layout.
5. Remove the obsolete `outputs_core/inputs/hcp7t_hippocampus` directory after
   validation confirms no active dependency remains.

Temporary compatibility links may be useful only during the migration window,
not as the final state.

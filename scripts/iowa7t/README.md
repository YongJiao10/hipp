# Iowa7T Utilities

These dataset adapters handle acquisition-specific anatomy, sampled signals,
cortical references, FC and figures. Run local Python commands with
`/home/yojiao/.conda/envs/py314/bin/python` from the repository root.

| Task | Entrypoint / guide |
|---|---|
| HippUnfold batch preparation | `run_iowa7t_hippunfold.py --help` |
| Cortex ArealMSHBM submission | [Shared Argon interface](../ArealMSHBM_Kong2022/README.md#iowa7t-variants) |
| Acquisition-level surface/cortex/FC stages | `run_branch_downstream.py --help` |
| Run-level surface/cortex/FC stages | `run_runwise_downstream.py --help` |
| Portable HCP151 template export | `export_hcp_templates.py --help`, [template guide](../../docs/hipp_parcellation_network/lss_hcp_templates.md) |
| Production paths and usage | [Iowa7T output layout](../../docs/iowa7t_36p_hipp_outputs_layout.md) |

Argon 36P inputs use
`/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P`;
production products use its `hipp_outputs/`, with qsub logs under
`hipp_outputs/runs/logs/<run_id>/`. Use the local Jianglab prefix only when
executing locally. Keep multiband/multi-echo and denoising variants distinct.

## Downstream stages

The two downstream adapters expose `surface`, `cortex`, `fc` and `all` stages.
For runwise processing, pass an explicit `--run-tag` naming the intended
`kong2022_runwise_<tag>/` tree. Runs remain distinct:
`run-01=ME01`, `run-02=ME02`, `run-03=MB03`, `run-04=MB04`.

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/iowa7t/run_runwise_downstream.py \
  --subject <subject> --variant 22p --run-tag <explicit_tag> --run all \
  --tsnr-input-template '<baseline_root>/{subject}_{variant}_{run}_{domain}_{hemi}.npz' \
  --stage all
```

Native T1w hippocampal meshes are registered to the declared MNI BOLD using
subject fMRIPrep MNI-to-T1w image composites and ANTs point mapping.
Mapped meshes are saved under `surface/raw/sampling_surfaces/`; native T1w
geometry defines physical smoothing and `space-T1w` metric filenames.
Surface/gate summaries must record `sampling_space=MNI152NLin6Asym` and
`smoothing_space=T1w`. Incompatible products require an explicit approved
`--overwrite` rebuild.

## Baseline tSNR and FC

Both adapters require `--tsnr-input-template`. Supported placeholders are
`{subject}`, `{hemi}`, `{domain}` (hipp/cortex), `{variant}`, `{branch}` and
`{run}` where supported. The NPZ format is defined in the
[core input guide](../core_workflow/README.md#existing-rest-tsnr-baseline-inputs).
Preserve acquisition-specific samples and variant-specific retained frames.

Per-run scores use original baseline mean divided by linearly detrended sample
SD on retained frames. All included runs must reach the inherited comparison
threshold 25 and have finite, nonconstant consumed analysis signals. Missing
baselines fail. This setting does not establish a universal participant-QC
cutoff; see [fMRI QC](../../docs/hipp_parcellation_network/fmri_qc.md).

A changed gate invalidates smoothing. FC rejects incompatible surface/cortex
gate metadata; use `--overwrite` for approved baseline/input replacement.
Adapters import maintained run-segment/TSV and label helpers from the core
workflow. Shared cortex inputs use the selected `hipp_outputs/runs/cache/`
root, while HCP references and template banks use their explicit formal
`pfm` paths. Existing production replacement follows the exact-path approval
rules.

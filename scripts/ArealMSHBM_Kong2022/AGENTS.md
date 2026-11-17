# Kong2022 ArealMSHBM Interface Rules

Apply these rules to dataset adapters and Argon launches in this directory.
Use [README.md](README.md) for exact dataset roots, sessions, controls and commands.

- Adapt paths and data only. Changes to the Kong2022_ArealMSHBM algorithm,
  model parameters, MATLAB internals or session semantics require an explicit
  user request.
- Extend `submit_kong2022_areal_mshbm.sh` and its shared task script for
  standard dataset presets. Do not add duplicate launchers in other dataset
  folders or the numbered workflows. The maintained Iowa7T runwise launcher
  is a separate, explicitly selected one-run-per-call interface.
- Keep input/output variants isolated: Stanford 36P, ESTT 23P/36P,
  and Iowa7T acquisition/denoising branches use their declared matching roots.
  ESTT and Iowa7T are separate datasets. Validate explicit variant/root agreement.
- JJ_TMS/Stanford source runs remain separate MATLAB sessions, each with one
  all-ones censor file. HCP7T splits its supplied concatenated CIFTI using
  exact physical-run metadata, validating subject, IDs 1–4, lengths and total
  frames. ESTT splits the selected variant's actual CIFTI frame count into
  four balanced sessions; do not borrow another variant's lengths.
- PDC/BANDA consume exact cohort-manifest CIFTIs from official source metadata.
  Preserve visits, run identities and the complete declared roster. Require
  `KEEP_TMP=1 OVERWRITE=0`; existing formal products fail closed.
- Do not add structural `surf.gii` requirements to JJ_TMS, Stanford or Iowa7T
  cortex-input adaptation. ESTT's existing materialization contract explicitly
  records subject fsLR32k surfaces and sulc assets.
- HCP7T uses `pfm_areal_mshbm/sub-<id>/`, retains task work under repository
  `tmp/`, and protects existing label/provenance files. ESTT retains its
  subject-level CBIG work and complete label/dlabel/materialization assets.
- Keep login-node work lightweight. Run MATLAB on compute nodes via qsub.
  Use the documented Argon runtime; do not add a local py314 dependency.
- Before a real smoke or batch submission, show the exact qsub command or
  use `DRY_RUN=1`. A full batch requires user approval and a successful real
  smoke with both hemisphere label files. Existing session authorization
  applies; do not ask for the same approval again.
- Retain temporary work unless its cleanup has explicit authorization for the
  exact paths. Production replacement and archive access follow root rules.

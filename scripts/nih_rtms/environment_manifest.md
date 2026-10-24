# NIH rTMS preprocessing runtime

## fMRIPrep finalization and exact FreeSurfer table

- Container: `/Shared/jianglab/0_scripts/fmriprep_tools/fmriprep_v2314_kw.sif`.
- Container interpreter: `/opt/conda/envs/fmriprep/bin/python`.
- CLI source: `/opt/conda/envs/fmriprep/lib/python3.10/site-packages/fmriprep/cli/run.py`.
- Required CLI SHA-256: `bba72ede85ed4f9d3a18b1f19eddab1ba8584a6b64d558fea219f51205f42304`.
- Canonical table: `/templateflow/tpl-fsaverage/tpl-fsaverage_dseg.tsv`, supplied by the declared shared TemplateFlow cache.
- Required table SHA-256: `be53e1cd18634bb4091297d5f6ec51e44626c0fa4d57030ebd63a3281fe34582`.
- Overlay: `fmriprep_overlay/sitecustomize.py`, mounted read-only at `/nih_rtms_fmriprep_runtime`.

This CLI requests all fsaverage `dseg.tsv` assets when publishing its two
FreeSurfer dictionaries. The shared cache contains 84 matching assets, so that
query returns a list. Converting the list to a filename fails after the
scientific workflow succeeds; the CLI's final `sys.exit` masks that exception
with status 1. The overlay resolves exactly this call to the declared canonical
table, validates both hashes and leaves other TemplateFlow calls unchanged.
It terminates with status 78 if either source contract fails. The launcher
records the overlay digest and validates the effective resolver before running.

`finalize_fmriprep_nih_rtms.sh SOURCE_ARRAY_JOB_ID` is the explicit finalization
repair for an array whose scientific workflows and HTML reports succeeded. It
validates the complete 44-subject roster, all 66 functional runs, paired volume
and CIFTI outputs, confounds and successful source logs/reports. It then publishes
the exact canonical table as `desc-aseg_dseg.tsv` and `desc-aparcaseg_dseg.tsv`.
Existing different files fail closed. Its scheduler log records the source
array, validated cohort counts, both output digests and finalization status.
No scientific workflow or shared software cache is rebuilt by that command.

## XCP-D container and source gate

- Execution context: Argon SGE; the launcher uses no host Python dependency.
- Container: `/Shared/jianglab/0_scripts/fmriprep_tools/xcp_d_latest.sif`.
- Observed workflow: XCP-D 0.10.0.
- Container interpreter: `/usr/local/miniconda/bin/python`.
- Patched upstream module: `/usr/local/miniconda/lib/python3.10/site-packages/nipype/utils/filemanip.py`.
- Required upstream module SHA-256: `b1235123a446b9735f2fca8b3148fa79fb72ad2b2efe6e73bf3cfa1b8eea7621`.
- Repository overlay: `scripts/nih_rtms/runtime/sitecustomize.py`, mounted read-only at `/nih_rtms_runtime` and loaded through container `PYTHONPATH`.

Each task records the overlay SHA-256 and checks the loaded patch before processing. A source mismatch or an
overlay initialization error terminates Python with status 78; changing the
container requires reviewing the source and updating this manifest and gate.
The SIF and Jianglab software installation are not modified.

## Shared QC publication

Nipype's timestamp-based copy checks, unlinks and recreates a destination.
Concurrent subjects can remove `desc-linc_qc.json` between another process's
existence check and stat, causing `ds_qc_metadata` to fail. The overlay handles
only these exact container destinations:

- `/data/xcpd_surf/restfmri/desc-linc_qc.json`
- `/data/xcpd/restfmri/desc-linc_qc.json`

For these two files it validates the source as a nonempty JSON dictionary,
writes the original bytes to a unique sibling, flushes them and atomically
replaces the canonical filename. Source permissions are preserved. A failed
publication retains and reports its staged file. Other Nipype copies use the
original implementation. Subject computations remain parallel.

The launcher retains per-subject scratch caches for resumable execution. Each
container pass is logged separately in the subject work root and a node failure
in that pass fails the launcher even if XCP-D returns status zero. Existing SGE
jobs retain their submitted script; submit a new job to use a launcher change.

## Concurrency check

Use an empty, dedicated directory beneath `<repository>/tmp/` as a test-only
`/data` mount. Never bind production outputs to this check. Account for the
parent and two workers within the current Argon capacity before running it.

```bash
repo_root=/old_Users/yojiao/hipp
check_dir="${repo_root}/tmp/nih_rtms_atomic_qc_check"
mkdir -p "${check_dir}"
singularity exec --cleanenv \
  --env PYTHONPATH=/nih_rtms_runtime \
  -B "${repo_root}/scripts/nih_rtms/runtime:/nih_rtms_runtime:ro" \
  -B "${check_dir}:/data" \
  /Shared/jianglab/0_scripts/fmriprep_tools/xcp_d_latest.sif \
  /usr/local/miniconda/bin/python /nih_rtms_runtime/check_atomic_qc.py
```

This exercises two real writer processes against both canonical QC filenames,
rejects malformed JSON without changing the destination, and verifies that
ordinary file copying still works. Test artifacts remain in the dedicated
repository temporary directory; cleanup requires explicit approval.

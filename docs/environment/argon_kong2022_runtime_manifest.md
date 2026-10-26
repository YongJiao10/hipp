# Argon Kong2022 runtime

Run the shared Kong2022 SGE launcher from `/old_Users/yojiao/hipp` with
`SYSTEM_NAME=argon`. This runtime is separate from the local HippUnfold stack.

| Component | Selected runtime |
| --- | --- |
| Python | `/Users/yojiao/miniconda3/envs/py313/bin/python`, with nibabel and NumPy |
| MATLAB | Existing `matlab/R2021a` module, invoked with `-singleCompThread` |
| Workbench | Existing 2.1.0 build, `/old_Users/yojiao/hipp/software/workbench_2.1.0/bin_linux64/wb_command` |
| Workbench commit | `99bf66b28572005f6419330d3f58b6262637412d` |
| CBIG | `/old_Users/yojiao/hipp/external/CBIG` |
| Native math threads | One per process, enforced by the shared task script |

Explicitly pass the Workbench path through the supported `WB_COMMAND` control.
The shared Workbench 2.2.1 binary cannot start on the Argon glibc 2.17 login
host; its Qt6 and newer glibc requirements are not satisfied. Do not change
the local HippUnfold Workbench selection to accommodate Argon.

Before submission, run the selected executable with `-version` and
`-file-information <exact-input-cifti> -only-number-of-maps`. Validate the
consumed CIFTI's cortical brain models and frame count against the declared
input contract. The HCP7T input has 32,492 vertices per cortical hemisphere;
physical run IDs, lengths and total frames come from its exact run metadata.

The existing launcher records `WB_COMMAND` in its generated job wrapper and
the task records it in output provenance. Keep `KEEP_TMP=1 OVERWRITE=0` for
HCP7T supplementation. Use the shared interface and concurrency limits in the
[launcher guide](../../scripts/ArealMSHBM_Kong2022/README.md).

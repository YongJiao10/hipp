"""Pin native math libraries to one thread. Import this module before numpy.

The thread count changes the last bits of LAPACK results, so a fixed count is
part of reproducibility as well as of the repository concurrency cap.
"""
import os

NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _name in NATIVE_THREAD_ENV_VARS:
    os.environ[_name] = "1"

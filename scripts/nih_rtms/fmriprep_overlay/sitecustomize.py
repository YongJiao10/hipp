"""Resolve fMRIPrep's final FreeSurfer table by its exact declared filename."""

import functools
import hashlib
import os
from pathlib import Path
import sys


CLI_SOURCE = Path("/opt/conda/envs/fmriprep/lib/python3.10/site-packages/fmriprep/cli/run.py")
CLI_SHA256 = "bba72ede85ed4f9d3a18b1f19eddab1ba8584a6b64d558fea219f51205f42304"
TABLE = Path("/templateflow/tpl-fsaverage/tpl-fsaverage_dseg.tsv")
TABLE_SHA256 = "be53e1cd18634bb4091297d5f6ec51e44626c0fa4d57030ebd63a3281fe34582"


def install():
    for path, expected in ((CLI_SOURCE, CLI_SHA256), (TABLE, TABLE_SHA256)):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source contract mismatch: {path}: expected {expected}, got {actual}")

    from templateflow import api

    original_get = api.get

    @functools.wraps(original_get)
    def get(*args, **kwargs):
        if args == ("fsaverage",) and kwargs == {"suffix": "dseg", "extension": [".tsv"]}:
            return TABLE
        return original_get(*args, **kwargs)

    get._nih_rtms_fsaverage_table = str(TABLE)
    get._nih_rtms_table_sha256 = TABLE_SHA256
    api.get = get


try:
    install()
except BaseException as error:
    sys.stderr.write(f"NIH rTMS fMRIPrep runtime gate failed: {error}\n")
    sys.stderr.flush()
    os._exit(78)

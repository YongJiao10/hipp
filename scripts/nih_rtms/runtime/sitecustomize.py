"""Atomic publication of XCP-D's two shared QC dictionaries.

Loaded only through the NIH rTMS container PYTHONPATH. All other Nipype
copies retain the container implementation. See ../environment_manifest.md.
"""

import functools
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


SOURCE_SHA256 = "b1235123a446b9735f2fca8b3148fa79fb72ad2b2efe6e73bf3cfa1b8eea7621"
QC_TARGETS = frozenset({
    "/data/xcpd_surf/restfmri/desc-linc_qc.json",
    "/data/xcpd/restfmri/desc-linc_qc.json",
})


def install():
    from nipype.utils import filemanip

    source = Path(filemanip.__file__)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"Unsupported Nipype source {source}: expected {SOURCE_SHA256}, got {actual}"
        )
    original_copyfile = filemanip.copyfile

    @functools.wraps(original_copyfile)
    def copyfile(originalfile, newfile, copy=False, create_new=False,
                 hashmethod=None, use_hardlink=False, copy_related_files=True):
        if os.path.abspath(os.fspath(newfile)) not in QC_TARGETS:
            return original_copyfile(
                originalfile, newfile, copy, create_new, hashmethod,
                use_hardlink, copy_related_files,
            )
        if create_new:
            raise ValueError(f"Shared QC metadata requires its exact filename: {newfile}")

        payload = Path(originalfile).read_bytes()
        metadata = json.loads(payload)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(f"Expected a nonempty QC metadata dictionary: {originalfile}")
        destination = Path(newfile)
        # A unique sibling permits atomic replacement on the output filesystem.
        # On failure, retain it and report its exact path; do not delete files.
        fd, staged = tempfile.mkstemp(
            prefix=".nih_rtms_atomic_qc_", dir=destination.parent,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), stat.S_IMODE(os.stat(originalfile).st_mode))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, destination)
        except BaseException:
            sys.stderr.write(f"NIH rTMS QC publication failed; staged file: {staged}\n")
            raise
        return newfile

    copyfile._nih_rtms_atomic_qc_copy = True
    copyfile._nih_rtms_source_sha256 = actual
    filemanip.copyfile = copyfile


try:
    install()
except BaseException as error:
    # Python normally ignores sitecustomize errors. This runtime must fail closed.
    sys.stderr.write(f"NIH rTMS runtime gate failed: {error}\n")
    sys.stderr.flush()
    os._exit(78)

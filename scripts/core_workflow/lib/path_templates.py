#!/usr/bin/env python3
"""Unified per-subject path-template formatting for core workflow entrypoints.

Use ``format_subject_path_template`` to resolve CLI-configurable path
templates instead of hard-coding HCP naming conventions.  This allows
non-HCP datasets (Stanford, Iowa, …) to supply their own file-name
patterns via ``--dtseries-template``, ``--bold-template``, and similar
arguments without a broad symlink tree.
"""

from __future__ import annotations

from pathlib import Path


def normalize_subject(subject: str) -> str:
    """Return the subject identifier without the ``sub-`` prefix."""
    return str(subject).replace("sub-", "", 1)


def format_subject_path_template(
    template: str,
    *,
    subject: str,
    hemi: str | None = None,
    **roots: str | Path,
) -> Path:
    """Format a per-subject path template.

    Supported variables
    -------------------
    - ``{subject}``      – identifier *without* ``sub-``, e.g. ``2001``
    - ``{sub_subject}``  – identifier *with* ``sub-``, e.g. ``sub-2001``
    - ``{hemi}``         – ``L`` or ``R`` when *hemi* is provided,
      otherwise an empty string
    - ``{hemi_lower}``   – ``l`` or ``r`` when *hemi* is provided,
      otherwise an empty string
    - any keyword argument passed in ``**roots``, resolved to an
      absolute path before substitution (e.g. ``data_root``,
      ``bold_root``)

    Raises
    ------
    KeyError
        If the template references a variable that is not listed above
        and was not supplied as a keyword argument.
    """
    subject_id = normalize_subject(subject)
    hemi_value = "" if hemi is None else str(hemi)
    values: dict[str, str] = {
        name: str(Path(value).resolve())
        for name, value in roots.items()
    }
    values.update(
        {
            "subject": subject_id,
            "sub_subject": f"sub-{subject_id}",
            "hemi": hemi_value,
            "hemi_lower": hemi_value.lower(),
        }
    )
    try:
        rendered = str(template).format(**values)
    except KeyError as exc:
        available = ", ".join(sorted(values))
        raise KeyError(
            f"Unknown path template variable {exc!s}; "
            f"available variables: {available}"
        ) from exc
    return Path(rendered).resolve()

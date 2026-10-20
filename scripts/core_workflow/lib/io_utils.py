from __future__ import annotations

import csv
import glob
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

_RUN_ID_RE = re.compile(r"(?:^|[_-])run-(?P<run>\d+)(?:[_\-.]|$)")
_PERSISTENT_CONCAT_COMPONENT_RE = re.compile(r"^xcpd_surf_concat$", re.IGNORECASE)


def is_persistent_concat_path(path: str | Path) -> bool:
    """Return whether a path is inside Iowa's legacy concat derivative tree.

    HCP's canonical single-file CIFTI may contain ``run-concat`` in its
    filename; the production prohibition is intentionally scoped to the
    generated ``xcpd_surf_concat`` directory used by Iowa JJ_TMS.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    return any(_PERSISTENT_CONCAT_COMPONENT_RE.match(part) for part in resolved.parts)


def _natural_path_sort_key(value: str | Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(chunk)) if chunk.isdigit() else (1, chunk.lower())
        for chunk in re.split(r"(\d+)", str(value))
    )


def run_id_from_path(path: str | Path, *, require_numeric: bool = True) -> int | None:
    """Return the numeric BIDS-style run ID encoded in a filename."""
    match = _RUN_ID_RE.search(Path(path).name)
    if match is None:
        if require_numeric:
            raise ValueError(f"Could not parse numeric run ID from path: {path}")
        return None
    return int(match.group("run"))


def run_ids_from_paths(
    paths: Sequence[str | Path], *, require_numeric: bool = True
) -> list[int]:
    """Return run IDs in the supplied path order, rejecting ambiguous paths."""
    run_ids = [
        run_id_from_path(path, require_numeric=require_numeric) for path in paths
    ]
    if any(run_id is None for run_id in run_ids):
        return []
    normalized = [int(run_id) for run_id in run_ids if run_id is not None]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate numeric run IDs in paths: {list(paths)}")
    return normalized


def resolve_run_paths_by_id(
    template: str,
    *,
    data_root: Path,
    subject: str,
    label: str = "run input",
) -> dict[int, Path]:
    """Resolve a per-subject glob and preserve the source numeric run IDs."""
    from .path_templates import format_subject_path_template

    pattern = format_subject_path_template(
        template,
        data_root=data_root,
        subject=subject,
    )
    matched = [
        Path(raw).resolve() for raw in glob.glob(str(pattern)) if Path(raw).is_file()
    ]
    if not matched:
        raise FileNotFoundError(
            f"No {label} files matched for sub-{str(subject).replace('sub-', '')}: {pattern}"
        )
    by_id: dict[int, Path] = {}
    for path in sorted(matched, key=_natural_path_sort_key):
        run_id = run_id_from_path(path)
        assert run_id is not None
        if run_id in by_id:
            raise ValueError(
                f"Duplicate run-{run_id:02d} {label} files: {by_id[run_id]} and {path}"
            )
        by_id[run_id] = path
    return dict(sorted(by_id.items()))


def resolve_run_dtseries_paths(
    template: str,
    *,
    data_root: Path,
    subject: str,
) -> list[Path]:
    """Resolve a per-subject run-dtseries glob without creating a concat file."""
    return list(
        resolve_run_paths_by_id(
            template,
            data_root=data_root,
            subject=subject,
            label="run dtseries",
        ).values()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_artifact_manifest(
    path: Path,
    *,
    subject: str,
    required_artifacts: Sequence[str],
) -> list[dict[str, object]]:
    """Load one caller-selected, fully receipted run-artifact manifest.

    The JSON list order is the selected run order.  This function deliberately
    does not glob, sort paths, or derive a run identifier from any filename.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing explicit run artifact manifest: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed run artifact manifest: {manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Run artifact manifest must be a JSON object: {manifest_path}"
        )

    subject_id = str(subject).strip().removeprefix("sub-")
    if not subject_id or payload.get("subject") != f"sub-{subject_id}":
        raise ValueError(f"Run artifact manifest subject mismatch: {manifest_path}")
    artifact_names = tuple(str(name) for name in required_artifacts)
    if not artifact_names or len(set(artifact_names)) != len(artifact_names):
        raise ValueError(
            "required_artifacts must be a nonempty duplicate-free sequence"
        )
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Run artifact manifest has no explicit runs: {manifest_path}")

    records: list[dict[str, object]] = []
    previous_run_id = 0
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, Mapping):
            raise ValueError(
                f"Run artifact manifest run {index} is not an object: {manifest_path}"
            )
        raw_run_id = raw_run.get("run_id")
        if (
            isinstance(raw_run_id, bool)
            or not isinstance(raw_run_id, int)
            or raw_run_id <= 0
        ):
            raise ValueError(
                "Run artifact manifest requires positive explicit run_id: "
                f"{manifest_path} run_index={index}"
            )
        run_id = int(raw_run_id)
        if run_id <= previous_run_id:
            raise ValueError(
                "Run artifact manifest run IDs must be strictly increasing: "
                f"{manifest_path}"
            )
        previous_run_id = run_id

        artifacts: dict[str, Path] = {}
        artifact_receipts: dict[str, dict[str, object]] = {}
        for artifact_name in artifact_names:
            recorded = raw_run.get(artifact_name)
            if not isinstance(recorded, Mapping):
                raise ValueError(
                    f"Run artifact manifest lacks {artifact_name!r} receipt "
                    f"for run-{run_id}: {manifest_path}"
                )
            raw_artifact_path = recorded.get("path")
            if (
                not isinstance(raw_artifact_path, str)
                or not raw_artifact_path
                or not Path(raw_artifact_path).is_absolute()
            ):
                raise ValueError(
                    f"Run artifact manifest requires an absolute {artifact_name!r} path "
                    f"for run-{run_id}: {manifest_path}"
                )
            declared_path = Path(raw_artifact_path)
            resolved_path = declared_path.resolve(strict=False)
            if declared_path != resolved_path:
                raise ValueError(
                    f"Run artifact manifest path must be canonical for "
                    f"{artifact_name!r} run-{run_id}: {declared_path}"
                )
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Missing declared {artifact_name!r} for run-{run_id}: {resolved_path}"
                )
            recorded_sha256 = recorded.get("sha256")
            recorded_size = recorded.get("size_bytes")
            if (
                not isinstance(recorded_sha256, str)
                or isinstance(recorded_size, bool)
                or not isinstance(recorded_size, int)
                or recorded_size < 0
            ):
                raise ValueError(
                    f"Run artifact manifest requires SHA-256 and size for "
                    f"{artifact_name!r} run-{run_id}: {manifest_path}"
                )
            actual_size = resolved_path.stat().st_size
            actual_sha256 = _sha256_file(resolved_path)
            if recorded_sha256 != actual_sha256 or recorded_size != actual_size:
                raise ValueError(
                    f"Run artifact receipt fingerprint drift for "
                    f"{artifact_name!r} run-{run_id}: {resolved_path}"
                )
            artifacts[artifact_name] = resolved_path
            artifact_receipts[artifact_name] = {
                "path": str(resolved_path),
                "sha256": actual_sha256,
                "size_bytes": int(actual_size),
            }
        records.append(
            {
                "run_id": run_id,
                "artifacts": artifacts,
                "artifact_receipts": artifact_receipts,
            }
        )
    return records

def load_run_segment_metadata(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Load explicit ordered run IDs and lengths from one selected JSON receipt."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing explicitly selected run metadata: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed run metadata JSON: {source}") from exc

    def integer(value: object) -> int:
        if isinstance(value, bool) or not (
            isinstance(value, int) or isinstance(value, str) and value.isdecimal()
        ):
            raise ValueError(f"Run metadata requires integer IDs and lengths: {source}")
        return int(value)

    def record_length(row: Mapping[str, object]) -> int | None:
        declared = [integer(row[key]) for key in (
            "run_length", "n_timepoints", "volume_timepoints", "dtseries_timepoints"
        ) if key in row]
        if not declared:
            return None
        if len(set(declared)) != 1:
            raise ValueError(f"Run metadata timepoints disagree: {source}")
        return declared[0]

    def parse(obj: object) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        if not isinstance(obj, Mapping):
            return None
        nested = obj.get("header_evidence")
        if isinstance(nested, Mapping):
            parsed = parse(nested)
            if parsed is not None:
                return parsed

        raw_lengths = obj.get("run_lengths")
        raw_ids = obj.get("run_ids")
        raw_runs = obj.get("run_segments", obj.get("runs"))
        records: list[Mapping[str, object]] | None = None
        if isinstance(raw_runs, list):
            if not all(isinstance(row, Mapping) for row in raw_runs):
                raise ValueError(f"Run metadata records must be objects: {source}")
            records = [row for row in raw_runs if isinstance(row, Mapping)]

        if isinstance(raw_lengths, list) and raw_lengths:
            lengths = tuple(integer(value) for value in raw_lengths)
            if records is not None:
                ids = tuple(integer(row["run_id"]) for row in records if "run_id" in row)
                lengths_from_records = tuple(
                    integer(record_length(row))
                    for row in records
                    if record_length(row) is not None
                )
                if len(ids) != len(records) or len(lengths_from_records) != len(records):
                    raise ValueError(f"Run metadata records lack explicit IDs or lengths: {source}")
                if lengths_from_records != lengths:
                    raise ValueError(f"run_lengths and runs disagree: {source}")
            elif isinstance(raw_ids, list) and len(raw_ids) == len(lengths):
                ids = tuple(integer(value) for value in raw_ids)
            else:
                raise ValueError(f"Run metadata lacks explicit run IDs matching lengths: {source}")
        elif records:
            if not all(
                "run_id" in row and record_length(row) is not None
                for row in records
            ):
                raise ValueError(f"Run metadata records lack explicit IDs or lengths: {source}")
            ids = tuple(integer(row["run_id"]) for row in records)
            lengths = tuple(
                integer(record_length(row)) for row in records
            )
        else:
            return None

        if raw_ids is not None and (
            not isinstance(raw_ids, list) or tuple(integer(value) for value in raw_ids) != ids
        ):
            raise ValueError(f"run_ids and run records disagree: {source}")
        if len(ids) != len(lengths) or not ids:
            raise ValueError(f"run_ids/run_lengths mismatch in run metadata: {source}")
        if any(run_id <= 0 for run_id in ids) or any(length <= 0 for length in lengths):
            raise ValueError(f"Invalid run ID or length in run metadata: {source}")
        if tuple(sorted(set(ids))) != ids:
            raise ValueError(f"Run IDs must be strictly increasing: {source}")
        declared_total = obj.get("concat_timepoints")
        if declared_total is not None and integer(declared_total) != sum(lengths):
            raise ValueError(f"Run metadata total does not match run lengths: {source}")
        return ids, lengths

    parsed = parse(payload)
    if parsed is None:
        raise ValueError(f"Run metadata contains no explicit run fields: {source}")
    return parsed


def build_run_level_input_summary(
    paths: Sequence[str | Path],
    run_lengths: Sequence[int],
    *,
    run_ids: Sequence[int],
    input_mode: str,
) -> dict[str, object]:
    """Summarize run-level inputs used for transient in-memory concatenation."""
    input_paths = [Path(path).expanduser().resolve() for path in paths]
    if not input_paths:
        raise ValueError("At least one run-level dtseries input is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in run_lengths
    ):
        raise ValueError(f"Run lengths must be positive integers: {run_lengths}")
    lengths = [int(value) for value in run_lengths]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in run_ids
    ):
        raise ValueError(f"Run IDs must be positive integers: {run_ids}")
    ids = [int(value) for value in run_ids]
    if len(lengths) != len(input_paths):
        raise ValueError(
            f"Run length/input count mismatch: lengths={lengths} inputs={input_paths}"
        )
    if len(ids) != len(input_paths):
        raise ValueError(
            f"Run ID/input count mismatch: run_ids={ids} inputs={input_paths}"
        )
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError(f"Run IDs must be strictly increasing: {ids}")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"Run lengths must be positive: {lengths}")
    mode = str(input_mode).strip()
    if not mode:
        raise ValueError("input_mode must not be empty")
    return {
        "dtseries_input_mode": mode,
        "dtseries_inputs": [str(path) for path in input_paths],
        "run_ids": ids,
        "run_lengths": lengths,
        "n_timepoints": int(sum(lengths)),
    }


def _cifti_axes_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    for name in ("name", "vertex", "voxel", "nvertices", "affine", "volume_shape"):
        left_value = getattr(left, name, None)
        right_value = getattr(right, name, None)
        if isinstance(left_value, dict) or isinstance(right_value, dict):
            if left_value != right_value:
                return False
            continue
        if not np.array_equal(np.asarray(left_value), np.asarray(right_value)):
            return False
    return True


def load_cifti_dtseries_inputs(
    paths: Sequence[str | Path],
) -> tuple[np.ndarray, Any, list[int]]:
    """Load and concatenate CIFTI dtseries runs in memory.

    The returned data are time-by-grayordinate and the source files remain the
    only on-disk inputs.  Every run must expose the same BrainModelAxis.
    """
    import nibabel as nib

    input_paths = [Path(path).resolve() for path in paths]
    if not input_paths:
        raise ValueError("At least one CIFTI dtseries input is required")

    data_blocks: list[np.ndarray] = []
    run_lengths: list[int] = []
    reference_axis: Any | None = None
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing CIFTI dtseries input: {path}")
        image = cast(Any, nib.load(str(path)))
        data = np.asarray(image.dataobj, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D CIFTI dtseries data: {path} shape={data.shape}"
            )
        axis = image.header.get_axis(1)
        if reference_axis is None:
            reference_axis = axis
        elif not _cifti_axes_equal(reference_axis, axis):
            raise ValueError(
                f"BrainModelAxis mismatch between CIFTI dtseries inputs: {input_paths[0]} vs {path}"
            )
        data_blocks.append(data)
        run_lengths.append(int(data.shape[0]))

    if reference_axis is None:  # pragma: no cover - guarded by non-empty input_paths
        raise RuntimeError("No BrainModelAxis loaded from CIFTI dtseries inputs")
    return np.concatenate(data_blocks, axis=0), reference_axis, run_lengths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_subject_id(raw: str) -> str:
    return str(raw).strip().replace("sub-", "")


def load_subject_ids(path: Path, *, require_nonempty: bool = True) -> list[str]:
    out: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        s = normalize_subject_id(line)
        if s:
            out.append(s)
    if require_nonempty and not out:
        raise RuntimeError(f"No subjects in manifest: {path}")
    return out


def compute_cortex_tsnr_gate(
    *,
    data: np.ndarray,
    outdir: Path,
    hemi: str,
    threshold: float | None,
    hcp_mode: float = 10000.0,
    task: str = "rest",
    run_lengths: Sequence[int] | None = None,
    dataset: str = "hcp",
    task_bold_paths: Sequence[str | Path] | None = None,
    tsnr_input: Path | None = None,
    subject: str | None = None,
    vertex_ids: np.ndarray | None = None,
) -> Path:
    from .task_quality import (
        parse_tsnr_threshold,
        rest_signal_validity,
        signal_validity_policy,
    )

    policy = signal_validity_policy(dataset, task)
    cutoff = parse_tsnr_threshold(threshold)
    if policy != "hcp_10000_over_std":
        from .task_design import normalize_task_branch
        from .tsnr import rest_tsnr_input, task_tsnr

        if task != "rest":
            task = normalize_task_branch(task)
            scores, valid = task_tsnr(data, [] if run_lengths is None else run_lengths,
                                      [] if task_bold_paths is None else task_bold_paths,
                                      task=task, threshold=cutoff)
        else:
            if tsnr_input is None or subject is None:
                raise ValueError(f"{dataset} cortex tSNR requires explicit baseline samples, subject and vertex order")
            scores, valid = rest_tsnr_input(tsnr_input, dataset=dataset, subject=subject,
                                            hemi="L" if hemi.lower() in ("left", "l") else "R",
                                            n_vertices=data.shape[0], n_frames=data.shape[1], threshold=threshold, vertex_ids=vertex_ids)
            valid &= rest_signal_validity(data)
        outdir.mkdir(parents=True, exist_ok=True)
        mask_path = outdir / f"cortex_{hemi.lower()}_valid_mask.npy"
        np.save(mask_path, valid)
        np.save(outdir / f"cortex_{hemi.lower()}_run_tsnr.npy", scores)
        np.save(outdir / f"cortex_{hemi.lower()}_tsnr.npy", np.min(scores, axis=0))
        task_validity_only = task != "rest" and cutoff is None
        summary = {
            "hemi": hemi, "task": task, "dataset": dataset, "validity_policy": policy,
            "baseline_input": str(tsnr_input) if tsnr_input is not None else None,
            "threshold": None if cutoff is None else float(cutoff),
            "tsnr_definition": policy,
            "run_aggregation": "validity_only_no_tsnr" if task_validity_only else "all_runs_pass",
            "run_lengths": list(run_lengths) if run_lengths is not None else None,
            "n_grayordinates_total": int(valid.size), "n_grayordinates_valid": int(valid.sum()),
            "n_grayordinates_invalid": int((~valid).sum()),
            "invalid_fraction": float((~valid).sum() / max(1, valid.size)),
        }
        if task != "rest":
            summary["tsnr_computed"] = cutoff is not None
        (outdir / f"cortex_{hemi.lower()}_tsnr_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return mask_path
    sd = np.nanstd(np.asarray(data, dtype=np.float64), axis=1, ddof=1)
    tsnr = np.where(sd > 0, hcp_mode / sd, np.nan).astype(np.float32)
    valid = (
        rest_signal_validity(data) if cutoff is None
        else np.isfinite(tsnr) & (tsnr >= float(cutoff))
    )
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"cortex_{hemi.lower()}_tsnr.npy", tsnr)
    mask_path = outdir / f"cortex_{hemi.lower()}_valid_mask.npy"
    np.save(mask_path, valid.astype(bool))

    # Write json directly
    gate_summary = {
        "hemi": hemi,
        "dataset": dataset,
        "task": task,
        "validity_policy": policy,
        "threshold": None if cutoff is None else float(cutoff),
        "tsnr_definition": f"{int(hcp_mode)}/std",
        "n_grayordinates_total": int(valid.size),
        "n_grayordinates_valid": int(valid.sum()),
        "n_grayordinates_invalid": int(valid.size - valid.sum()),
        "invalid_fraction": float(1.0 - (valid.sum() / max(1, valid.size))),
    }
    summary_path = outdir / f"cortex_{hemi.lower()}_tsnr_gate_summary.json"
    summary_path.write_text(json.dumps(gate_summary, indent=2) + "\n", encoding="utf-8")
    return mask_path


def parse_run_segments(
    run_lengths: Sequence[int] | None,
    expected_total_tp: int,
    *,
    run_ids: Sequence[int] | None,
) -> list[dict[str, object]]:
    if run_ids is None:
        raise ValueError("parse_run_segments requires explicit run IDs")
    if run_lengths is None:
        raise ValueError("parse_run_segments requires explicit run lengths")

    ids = list(run_ids)
    lengths = list(run_lengths)
    if not ids:
        raise ValueError("parse_run_segments requires at least one explicit run ID")
    if len(lengths) != len(ids):
        raise ValueError(f"Expected {len(ids)} run lengths, got {lengths}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in ids
    ):
        raise ValueError(f"Run IDs must be positive integers: {ids}")
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError(f"Run IDs must be strictly increasing: {ids}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in lengths
    ):
        raise ValueError(f"Run lengths must be positive integers: {lengths}")

    segments: list[dict[str, object]] = []
    offset = 0
    for rid, run_len in zip(ids, lengths, strict=True):
        run_start = int(offset)
        run_stop = int(offset + run_len)
        segments.append(
            {
                "run_id": int(rid),
                "run_label": f"run-{rid}",
                "run_start": run_start,
                "run_stop": run_stop,
                "run_length": int(run_len),
            }
        )
        offset = run_stop
    if offset != int(expected_total_tp):
        raise RuntimeError(
            f"Run lengths do not sum to concat length: sum={offset} concat={expected_total_tp}"
        )
    return segments


def validate_run_segment_order(
    segments: list[dict[str, int]],
    *,
    source: Path,
) -> list[dict[str, int]]:
    if not segments:
        raise ValueError(f"Run metadata is empty: {source}")
    out: list[dict[str, int]] = []
    previous_id = 0
    previous_stop = 0
    for index, raw in enumerate(segments, start=1):
        try:
            run_id = int(raw["run_id"])
            start = int(raw["run_start"])
            stop = int(raw["run_stop"])
            length = int(raw["run_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed run segment at index {index}: {raw}; source={source}") from exc
        if run_id <= previous_id:
            raise ValueError(f"Run IDs must be strictly increasing: {source}")
        if index == 1 and start != 0:
            raise ValueError(f"Run segments must start at frame 0: {source}")
        if start != previous_stop:
            raise ValueError(f"Run segments must be contiguous: {source}")
        if start < 0 or stop <= start or length != stop - start:
            raise ValueError(f"Invalid run segment: {raw}; source={source}")
        out.append({"run_id": run_id, "run_start": start, "run_stop": stop, "run_length": length})
        previous_id = run_id
        previous_stop = stop
    return out


def segments_from_explicit_metadata(path: Path) -> list[dict[str, int]]:
    resolved = Path(path).resolve()
    run_ids, run_lengths = load_run_segment_metadata(resolved)
    segments: list[dict[str, int]] = []
    offset = 0
    for run_id, run_length in zip(run_ids, run_lengths, strict=True):
        stop = offset + int(run_length)
        segments.append({"run_id": int(run_id), "run_start": int(offset), "run_stop": int(stop), "run_length": int(run_length)})
        offset = stop
    return validate_run_segment_order(segments, source=resolved)


def load_optional_cortex_valid_mask(
    reference_root: Path,
    subject: str,
    cortex_counts: Mapping[str, int],
    *,
    require_explicit: bool = True,
) -> np.ndarray:
    """Load explicit cortical validity masks; no implicit mask is default."""
    subject = str(subject).replace("sub-", "")
    gate = Path(reference_root) / f"sub-{subject}" / "reference" / "common" / "tsnr_gate"
    pieces: list[np.ndarray] = []
    missing: list[Path] = []
    for hemi_name, key in [("left", "CORTEX_LEFT"), ("right", "CORTEX_RIGHT")]:
        path = gate / f"cortex_{hemi_name}_valid_mask.npy"
        if path.is_file():
            arr = np.load(path).astype(bool)
            if int(arr.size) != int(cortex_counts[key]):
                raise ValueError(f"Cortex valid mask size mismatch: {path} size={arr.size} expected={cortex_counts[key]}")
            pieces.append(arr)
        else:
            missing.append(path)
            if not require_explicit:
                pieces.append(np.ones((int(cortex_counts[key]),), dtype=bool))
    if missing and require_explicit:
        raise FileNotFoundError(
            "Missing explicit cortical validity mask(s) required for cortex/FC processing: "
            + ", ".join(str(path) for path in missing)
        )
    return np.concatenate(pieces).astype(bool)

"""Strict provenance checks shared by fixed-template overview renderers.

The contract deliberately receives every source path from the caller.  It does
not discover roster files, matching runs, scenes, or output locations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


def fingerprint(path: Path) -> dict[str, object]:
    """Return a complete SHA-256 record for one exact regular file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing fingerprint input: {resolved}")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(size),
    }


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed {context}: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {context} object: {resolved}")
    return payload


def _validate_fingerprint(
    recorded: object,
    path: Path,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Missing {context} fingerprint: {path}")
    actual = fingerprint(path)
    if dict(recorded) != actual:
        raise ValueError(f"{context} fingerprint drift: {path}")
    return actual


def _normalize_subject(value: object, *, context: str) -> str:
    subject = str(value).strip().removeprefix("sub-")
    if not subject or not subject.isdecimal():
        raise ValueError(f"Invalid subject in {context}: {value!r}")
    return subject


def _load_roster(path: Path) -> tuple[str, ...]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing explicit roster: {resolved}")
    subjects = tuple(
        _normalize_subject(raw, context=f"roster {resolved}")
        for raw in resolved.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    )
    if not subjects:
        raise ValueError(f"Explicit roster is empty: {resolved}")
    return subjects


def _strict_count(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {context}: {value!r}")
    return value


@dataclass(frozen=True)
class RenderPreflight:
    """Exact read-only sources required before a renderer may create outputs."""

    subjects: tuple[str, ...]
    matching_root: Path
    matching_run_manifest: Mapping[str, object]
    roster_fingerprints: tuple[Mapping[str, object], ...]
    scene: Path
    scene_fingerprint: Mapping[str, object]


def preflight_render_inputs(
    *,
    matching_root: Path,
    run_manifest: Path,
    roster_paths: Sequence[Path],
    scene: Path,
    expected_matching_workflow: str,
    expected_subject_count: int,
) -> RenderPreflight:
    """Lock explicit Stanford-style roster, matching run, and scene inputs.

    This completes before rendering begins, so an incomplete or crossed cohort
    cannot leave a partial overview tree behind.
    """
    if not roster_paths:
        raise ValueError("At least one explicit roster path is required")
    if expected_subject_count <= 0:
        raise ValueError("expected_subject_count must be positive")
    matching = Path(matching_root).resolve()
    roster_records = tuple(fingerprint(Path(path)) for path in roster_paths)
    subjects = tuple(subject for path in roster_paths for subject in _load_roster(Path(path)))
    if len(subjects) != expected_subject_count or len(set(subjects)) != len(subjects):
        raise ValueError(
            "Explicit roster closure is invalid: "
            f"count={len(subjects)} expected={expected_subject_count}"
        )

    manifest_path = Path(run_manifest).resolve()
    payload = _load_json_object(manifest_path, context="matching run manifest")
    if payload.get("status") != "complete" or payload.get("scope") != "full":
        raise ValueError(f"Matching run is not a complete full run: {manifest_path}")
    if payload.get("workflow") != expected_matching_workflow:
        raise ValueError(
            "Matching run workflow mismatch: "
            f"expected={expected_matching_workflow!r} actual={payload.get('workflow')!r}"
        )
    for key in ("canonical_roster_count", "subject_count", "complete_count"):
        if _strict_count(payload.get(key), context=f"matching run {key}") != expected_subject_count:
            raise ValueError(
                f"Matching run {key} does not match explicit roster: {manifest_path}"
            )
    if _strict_count(payload.get("failed_count"), context="matching run failed_count") != 0:
        raise ValueError(f"Matching run has failures: {manifest_path}")
    recorded_subjects = payload.get("subjects")
    expected_subjects = [f"sub-{subject}" for subject in subjects]
    if not isinstance(recorded_subjects, list) or recorded_subjects != expected_subjects:
        raise ValueError(f"Matching run roster closure drift: {manifest_path}")
    path_roots = payload.get("path_roots")
    if not isinstance(path_roots, Mapping):
        raise ValueError(f"Matching run path_roots are missing: {manifest_path}")
    if Path(str(path_roots.get("out_root", ""))).resolve() != matching:
        raise ValueError(f"Matching run output root mismatch: {manifest_path}")
    template_record = payload.get("template_manifest")
    if not isinstance(template_record, Mapping):
        raise ValueError(f"Matching run template manifest record is missing: {manifest_path}")
    template_path = Path(str(template_record.get("path", ""))).resolve()
    _validate_fingerprint(template_record, template_path, context="matching template manifest")

    scene_path = Path(scene).resolve()
    scene_record = fingerprint(scene_path)
    return RenderPreflight(
        subjects=subjects,
        matching_root=matching,
        matching_run_manifest=fingerprint(manifest_path),
        roster_fingerprints=roster_records,
        scene=scene_path,
        scene_fingerprint=scene_record,
    )


def validate_existing_receipt(
    *,
    receipt_path: Path,
    preflight: RenderPreflight,
    expected_overview_workflow: str,
    output_dir: Path,
) -> dict[str, object]:
    """Require an existing overview receipt to match the current exact inputs."""
    current_scene_fingerprint = fingerprint(preflight.scene)
    if current_scene_fingerprint != dict(preflight.scene_fingerprint):
        raise ValueError(f"Existing overview scene drift: {receipt_path}")
    receipt = _load_json_object(receipt_path, context="overview receipt")
    if receipt.get("status") != "complete":
        raise ValueError(f"Existing overview receipt is not complete: {receipt_path}")
    if receipt.get("workflow") != expected_overview_workflow:
        raise ValueError(f"Existing overview workflow mismatch: {receipt_path}")
    if Path(str(receipt.get("output_dir", ""))).resolve() != Path(output_dir).resolve():
        raise ValueError(f"Existing overview output directory drift: {receipt_path}")
    expected_subjects = [f"sub-{subject}" for subject in preflight.subjects]
    if receipt.get("subjects") != expected_subjects or receipt.get("subject_count") != len(expected_subjects):
        raise ValueError(f"Existing overview roster closure drift: {receipt_path}")
    if receipt.get("matching_run_manifest") != dict(preflight.matching_run_manifest):
        raise ValueError(f"Existing overview matching-run drift: {receipt_path}")
    if receipt.get("scene") != str(preflight.scene):
        raise ValueError(f"Existing overview scene drift: {receipt_path}")
    if receipt.get("scene_fingerprint") != dict(preflight.scene_fingerprint):
        raise ValueError(f"Existing overview scene drift: {receipt_path}")
    return receipt


def _artifact_path(
    recorded: object,
    *,
    context: str,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Resolve one manifest-declared artifact and verify its core receipt."""
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Missing {context} artifact record")
    raw_path = recorded.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Missing {context} artifact path")
    resolved = Path(raw_path).resolve()
    if expected_path is not None and resolved != Path(expected_path).resolve():
        raise ValueError(
            f"{context} path mismatch: expected={Path(expected_path).resolve()} "
            f"actual={resolved}"
        )
    actual = fingerprint(resolved)
    if any(recorded.get(key) != actual[key] for key in actual):
        raise ValueError(f"{context} fingerprint drift: {resolved}")
    return resolved, actual


def _selected_values(values: Sequence[str], *, context: str) -> tuple[str, ...]:
    selected = tuple(str(value) for value in values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{context} must be a nonempty, duplicate-free explicit sequence")
    return selected


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer: {value!r}")
    return value


def preflight_subject_sources(
    *,
    preflight: RenderPreflight,
    expected_matching_workflow: str,
    expected_label_semantics: Mapping[str, str],
    smoothings: Sequence[str],
    hemis: Sequence[str],
    expected_n_vertices: int,
) -> tuple[dict[str, object], ...]:
    """Lock each manifest-declared subject source before a renderer can write.

    The caller provides the roster, matching run, and every semantic choice.
    This function never discovers alternate artifacts or creates output paths.
    """
    selected_smoothings = _selected_values(smoothings, context="smoothings")
    selected_hemis = _selected_values(hemis, context="hemis")
    n_vertices = _positive_int(
        expected_n_vertices,
        context="expected_n_vertices",
    )
    expected_semantics = {
        str(code): str(name) for code, name in expected_label_semantics.items()
    }
    if not expected_semantics:
        raise ValueError("expected_label_semantics must not be empty")

    run_manifest_path, _ = _artifact_path(
        preflight.matching_run_manifest,
        context="matching run manifest",
    )
    run_manifest = _load_json_object(
        run_manifest_path,
        context="matching run manifest",
    )
    expected_subjects = [f"sub-{subject}" for subject in preflight.subjects]
    if (
        run_manifest.get("status") != "complete"
        or run_manifest.get("scope") != "full"
        or run_manifest.get("workflow") != expected_matching_workflow
        or run_manifest.get("subjects") != expected_subjects
    ):
        raise ValueError(f"Matching run contract drift: {run_manifest_path}")
    template_record = run_manifest.get("template_manifest")
    template_path, template_fingerprint = _artifact_path(
        template_record,
        context="matching template manifest",
    )

    expected_combinations = {
        (smoothing, hemi)
        for smoothing in selected_smoothings
        for hemi in selected_hemis
    }
    source_rows: list[dict[str, object]] = []
    for subject in preflight.subjects:
        subject_dir = (
            preflight.matching_root / "subjects" / f"sub-{subject}"
        ).resolve()
        subject_manifest_path = subject_dir / "subject_manifest.json"
        subject_manifest = _load_json_object(
            subject_manifest_path,
            context=f"subject manifest sub-{subject}",
        )
        if (
            subject_manifest.get("status") != "ok"
            or subject_manifest.get("workflow") != expected_matching_workflow
            or subject_manifest.get("subject") != f"sub-{subject}"
        ):
            raise ValueError(f"Subject manifest identity mismatch: {subject_manifest_path}")
        subject_template = subject_manifest.get("template_manifest")
        subject_template_path, subject_template_fingerprint = _artifact_path(
            subject_template,
            context=f"subject template manifest sub-{subject}",
        )
        if (
            subject_template_path != template_path
            or subject_template_fingerprint != template_fingerprint
        ):
            raise ValueError(f"Subject template provenance drift: {subject_manifest_path}")

        inputs = subject_manifest.get("inputs")
        outputs = subject_manifest.get("outputs")
        combinations = subject_manifest.get("combinations")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            raise ValueError(f"Subject manifest inputs/outputs are missing: {subject_manifest_path}")
        if not isinstance(combinations, list):
            raise ValueError(f"Subject manifest combinations are missing: {subject_manifest_path}")

        combination_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
        for raw_combination in combinations:
            if not isinstance(raw_combination, Mapping):
                raise ValueError(f"Invalid subject combination: {subject_manifest_path}")
            key = (
                str(raw_combination.get("smoothing", "")),
                str(raw_combination.get("hemi", "")),
            )
            if key in combination_by_key:
                raise ValueError(
                    f"Duplicate subject combination sub-{subject}: {key}"
                )
            combination_by_key[key] = raw_combination
        if set(combination_by_key) != expected_combinations:
            raise ValueError(
                f"Subject combination closure drift sub-{subject}: "
                f"expected={sorted(expected_combinations)} "
                f"actual={sorted(combination_by_key)}"
            )

        surfaces: dict[str, dict[str, object]] = {}
        for hemi in selected_hemis:
            surface_record = inputs.get(f"surface_hemi-{hemi}")
            surface_path, surface_fingerprint = _artifact_path(
                surface_record,
                context=f"subject surface sub-{subject} hemi-{hemi}",
            )
            expected_surface_name = (
                f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
            if surface_path.name != expected_surface_name:
                raise ValueError(
                    f"Subject surface identity mismatch sub-{subject} hemi-{hemi}: "
                    f"{surface_path}"
                )
            surfaces[hemi] = {
                "path": str(surface_path),
                "fingerprint": surface_fingerprint,
            }

        subject_source_rows: list[dict[str, object]] = []
        for smoothing in selected_smoothings:
            source_row: dict[str, object] = {"smoothing": smoothing}
            for hemi in selected_hemis:
                key = (smoothing, hemi)
                combination = combination_by_key[key]
                combo_path = (
                    subject_dir
                    / "matching"
                    / smoothing
                    / f"hemi-{hemi}"
                    / "combo_manifest.json"
                ).resolve()
                raw_combo_path = combination.get("manifest")
                if (
                    not isinstance(raw_combo_path, str)
                    or Path(raw_combo_path).resolve() != combo_path
                ):
                    raise ValueError(
                        f"Subject combo manifest path mismatch sub-{subject} {key}"
                    )
                prefix = f"{smoothing}_hemi-{hemi}"
                combo_record = outputs.get(f"{prefix}_combo_manifest")
                resolved_combo_path, combo_fingerprint = _artifact_path(
                    combo_record,
                    context=f"combo manifest sub-{subject} {smoothing} hemi-{hemi}",
                    expected_path=combo_path,
                )
                combination_outputs = combination.get("outputs")
                if (
                    not isinstance(combination_outputs, Mapping)
                    or combination_outputs.get("combo_manifest") != combo_record
                ):
                    raise ValueError(
                        f"Subject combo provenance drift sub-{subject} {key}"
                    )
                combo_manifest = _load_json_object(
                    resolved_combo_path,
                    context=f"combo manifest sub-{subject} {smoothing} hemi-{hemi}",
                )
                if (
                    combo_manifest.get("status") != "ok"
                    or combo_manifest.get("workflow") != expected_matching_workflow
                    or combo_manifest.get("subject") != f"sub-{subject}"
                    or combo_manifest.get("smoothing") != smoothing
                    or combo_manifest.get("hemi") != hemi
                    or combo_manifest.get("label_semantics") != expected_semantics
                ):
                    raise ValueError(
                        f"Combo manifest identity mismatch sub-{subject} {key}"
                    )
                combo_outputs = combo_manifest.get("outputs")
                if not isinstance(combo_outputs, Mapping):
                    raise ValueError(
                        f"Combo manifest outputs are missing sub-{subject} {key}"
                    )
                label_record = combo_outputs.get("label_gifti")
                label_path, label_fingerprint = _artifact_path(
                    label_record,
                    context="label GIFTI",
                )
                expected_label_path = (
                    combo_path.parent
                    / (
                        f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                        f"label-hcp151_fixed_template_k3_{smoothing}.label.gii"
                    )
                ).resolve()
                if label_path != expected_label_path:
                    raise ValueError(
                        f"Label GIFTI path mismatch sub-{subject} {key}: {label_path}"
                    )
                subject_label_record = outputs.get(f"{prefix}_label_gifti")
                if (
                    not isinstance(subject_label_record, Mapping)
                    or dict(subject_label_record) != dict(label_record)
                    or combination_outputs.get("label_gifti") != label_record
                ):
                    raise ValueError(
                        f"Subject label provenance drift sub-{subject} {key}"
                    )

                density = combo_manifest.get("density_evidence")
                if not isinstance(density, Mapping):
                    raise ValueError(
                        f"Combo density evidence is missing sub-{subject} {key}"
                    )
                raw_surface = density.get("consumed_asset")
                if (
                    not isinstance(raw_surface, str)
                    or Path(raw_surface).resolve() != Path(
                        str(surfaces[hemi]["path"])
                    ).resolve()
                ):
                    raise ValueError(
                        f"Combo consumed surface drift sub-{subject} {key}"
                    )
                if _positive_int(
                    density.get("n_vertices"),
                    context=f"combo n_vertices sub-{subject} {key}",
                ) != n_vertices:
                    raise ValueError(
                        f"Combo vertex count mismatch sub-{subject} {key}"
                    )
                _artifact_path(
                    density.get("surface_fingerprint"),
                    context=f"combo consumed surface sub-{subject} {key}",
                    expected_path=Path(str(surfaces[hemi]["path"])),
                )
                raw_spec_path = density.get("spec_path")
                if not isinstance(raw_spec_path, str):
                    raise ValueError(f"Combo spec path is missing sub-{subject} {key}")
                spec_path = Path(raw_spec_path).resolve()
                expected_spec_name = (
                    f"sub-{subject}_den-512_label-hipp_surfaces.spec"
                )
                if spec_path.name != expected_spec_name:
                    raise ValueError(
                        f"Combo spec identity mismatch sub-{subject} {key}: {spec_path}"
                    )
                _, spec_fingerprint = _artifact_path(
                    density.get("spec_fingerprint"),
                    context=f"combo spec sub-{subject} {key}",
                    expected_path=spec_path,
                )
                subject_spec_record = inputs.get(f"surface_spec_hemi-{hemi}")
                if subject_spec_record is not None:
                    subject_spec_path, subject_spec_fingerprint = _artifact_path(
                        subject_spec_record,
                        context=f"subject spec sub-{subject} hemi-{hemi}",
                        expected_path=spec_path,
                    )
                    if subject_spec_path != spec_path or subject_spec_fingerprint != spec_fingerprint:
                        raise ValueError(
                            f"Subject spec provenance drift sub-{subject} {key}"
                        )
                if (
                    combination.get("density_evidence") != density
                    or combination_outputs.get("label_gifti") != label_record
                ):
                    raise ValueError(
                        f"Subject combination evidence drift sub-{subject} {key}"
                    )
                source_row[hemi] = {
                    "combo_manifest": str(resolved_combo_path),
                    "combo_manifest_fingerprint": combo_fingerprint,
                    "label_gifti": str(label_path),
                    "label_fingerprint": label_fingerprint,
                    "surface": {
                        **surfaces[hemi],
                        "spec_path": str(spec_path),
                        "spec_fingerprint": spec_fingerprint,
                        "n_vertices": n_vertices,
                    },
                }
            subject_source_rows.append(source_row)
        source_rows.append(
            {
                "subject": f"sub-{subject}",
                "subject_manifest": str(subject_manifest_path),
                "subject_manifest_fingerprint": fingerprint(subject_manifest_path),
                "source_rows": subject_source_rows,
            }
        )
    return tuple(source_rows)

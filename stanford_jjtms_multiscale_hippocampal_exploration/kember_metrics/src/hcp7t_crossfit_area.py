"""Build a native-area reference table from held-out HCP 7T labels.

The HCP 7T cohort supplied the template used by the Stanford matching branch.
Consequently, an HCP subject cannot be evaluated with the full HCP template
that helped define its own label map.  This module consumes the maintained
family-aware five-fold cross-fit subject labels instead, transfers each held-
out 512-source label map to that subject's native surface, and sums the same
native HippUnfold ``surfareacorobl`` metric used by the Stanford Kember table.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:  # Imports work as a package and from ``python src/script.py``.
    from .native_label_alignment import align_labels_to_native
    from .surface_area import compute_system_surface_area
    from .surface_geometry import load_midthickness
except ImportError:  # pragma: no cover - direct source-directory execution
    from native_label_alignment import align_labels_to_native  # type: ignore[no-redef]
    from surface_area import compute_system_surface_area  # type: ignore[no-redef]
    from surface_geometry import load_midthickness  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
HCP_GROUP = "HCP 7T"
HCP_METHOD = "hcp151_crossfit_template_matching"
HCP_SMOOTHING_MM = 2
EXPECTED_HCP_SUBJECTS = 151
EXPECTED_HCP_FOLDS = 5
EXPECTED_HCP_TEMPLATE_FITS = 20
SYSTEMS = ("anterior", "middle", "posterior")
HEMISPHERES = ("L", "R")
DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "work" / "reference_hcp7t_crossfit_area"
_CACHE_SCHEMA_VERSION = 2


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required HCP cross-fit manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"HCP cross-fit manifest must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _normalise_subject(value: Any) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw.isdigit():
        raise ValueError(f"invalid HCP subject identifier: {value!r}")
    return f"sub-{int(raw):06d}"


def validate_crossfit_run_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require the maintained family-aware five-fold HCP contract."""

    if int(manifest.get("n_subjects", -1)) != EXPECTED_HCP_SUBJECTS:
        raise ValueError("HCP cross-fit run must contain 151 subjects")
    if int(manifest.get("n_folds", -1)) != EXPECTED_HCP_FOLDS:
        raise ValueError("HCP cross-fit run must use five-fold cross-fitting")
    if int(manifest.get("template_fits_completed", -1)) != EXPECTED_HCP_TEMPLATE_FITS:
        raise ValueError("HCP cross-fit run must contain 20 completed template fits")
    family = manifest.get("family")
    if not isinstance(family, Mapping) or family.get("family_aware") is not True:
        raise ValueError("HCP cross-fit run must be family-aware")
    if family.get("subject_only_waiver") is not False:
        raise ValueError("HCP cross-fit run cannot use the subject-only waiver")
    return dict(manifest)


def validate_crossfit_combo_manifest(
    manifest: Mapping[str, Any],
    *,
    subject: str,
    hemisphere: str,
    smoothing_mm: int,
) -> dict[str, Any]:
    """Validate one subject combo as a genuinely held-out cross-fit result."""

    if manifest.get("status") != "ok":
        raise ValueError(f"HCP cross-fit combo is not successful: {subject} {hemisphere}")
    if _normalise_subject(manifest.get("subject")) != _normalise_subject(subject):
        raise ValueError("HCP cross-fit combo subject does not match its path")
    if str(manifest.get("hemi")) != hemisphere:
        raise ValueError("HCP cross-fit combo hemisphere does not match its path")
    if str(manifest.get("smoothing")) != f"{int(smoothing_mm)}mm":
        raise ValueError("HCP cross-fit combo smoothing does not match its path")
    if manifest.get("subject_was_in_template") is not False:
        raise ValueError("HCP subject must be held out from its template")
    training_subjects = {
        _normalise_subject(value) for value in manifest.get("template_training_subjects", ())
    }
    if _normalise_subject(subject) in training_subjects:
        raise ValueError("HCP subject must be held out from template training subjects")
    if manifest.get("template_fold_id") not in {f"fold-{index:02d}" for index in range(1, 6)}:
        raise ValueError("HCP cross-fit combo lacks a valid five-fold template id")
    semantics = manifest.get("label_semantics")
    if not isinstance(semantics, Mapping) or {
        str(key): str(value) for key, value in semantics.items()
    } != {"1": "Anterior", "2": "Middle", "3": "Posterior"}:
        raise ValueError("HCP cross-fit combo lacks the AP label semantics contract")
    return dict(manifest)


def _load_vector(path: Path, *, dtype: Any, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required HCP cross-fit {label} is missing: {path}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=dtype).reshape(-1)
    if not values.size:
        raise ValueError(f"HCP cross-fit {label} is empty: {path}")
    return values


def _load_gifti_vector(path: Path, *, dtype: Any, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required HCP native {label} is missing: {path}")
    image = nib.load(str(path))
    if not hasattr(image, "darrays") or len(image.darrays) != 1:
        raise ValueError(f"HCP native {label} must be a single-array GIFTI: {path}")
    values = np.asarray(image.darrays[0].data, dtype=dtype).reshape(-1)
    if not values.size:
        raise ValueError(f"HCP native {label} is empty: {path}")
    return values


def _surface_path(root: Path, subject: str, hemisphere: str, *, space: str, density: str) -> Path:
    return (
        root
        / subject
        / "hippunfold"
        / subject
        / "surf"
        / f"{subject}_hemi-{hemisphere}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii"
    )


def _native_area_path(root: Path, subject: str, hemisphere: str) -> Path:
    return (
        root
        / subject
        / "hippunfold"
        / subject
        / "metric"
        / f"{subject}_hemi-{hemisphere}_den-native_label-hipp_surfareacorobl.shape.gii"
    )


def _resolve_workbench(value: str | Path | None) -> Path:
    if value is None:
        discovered = shutil.which("wb_command")
        if discovered is None:
            raise FileNotFoundError("Connectome Workbench executable is required for HCP native label transfer: wb_command")
        candidate = Path(discovered)
    else:
        candidate = Path(value).expanduser()
    if not candidate.is_file() or not candidate.stat().st_mode & 0o111:
        raise FileNotFoundError(
            "Connectome Workbench executable is required for HCP native label transfer: "
            f"{candidate}"
        )
    return candidate.resolve()


def _reference_spec(config: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = config.get("reference_hcp7t")
    if not isinstance(spec, Mapping) or spec.get("enabled", True) is not True:
        raise ValueError("analysis config must enable the HCP 7T reference branch")
    return spec


def _subject_list(run_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    run_config = run_manifest.get("run_config")
    subjects = run_config.get("subjects") if isinstance(run_config, Mapping) else None
    if not isinstance(subjects, list):
        raise ValueError("HCP cross-fit manifest lacks run_config.subjects")
    normalised = tuple(_normalise_subject(value) for value in subjects)
    if len(normalised) != EXPECTED_HCP_SUBJECTS or len(set(normalised)) != len(normalised):
        raise ValueError("HCP cross-fit subject list must contain 151 unique subjects")
    return tuple(sorted(normalised, key=lambda value: int(value[4:])))


def _combo_specs(
    crossfit_root: Path,
    subjects: tuple[str, ...],
    *,
    smoothing_mm: int,
) -> list[tuple[str, str, Path, dict[str, Any]]]:
    result: list[tuple[str, str, Path, dict[str, Any]]] = []
    for subject in subjects:
        for hemisphere in HEMISPHERES:
            combo_path = (
                crossfit_root
                / "subjects"
                / subject
                / f"{smoothing_mm}mm"
                / f"hemi_{hemisphere}"
                / "combo_manifest.json"
            )
            combo = validate_crossfit_combo_manifest(
                _load_json(combo_path),
                subject=subject,
                hemisphere=hemisphere,
                smoothing_mm=smoothing_mm,
            )
            result.append((subject, hemisphere, combo_path, combo))
    return result


def _build_hcp_entry(
    *,
    subject: str,
    hemisphere: str,
    smoothing_mm: int,
    combo: Mapping[str, Any],
    crossfit_root: Path,
    hippunfold_root: Path,
    shared_surface_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subject_dir = crossfit_root / "subjects" / subject / f"{smoothing_mm}mm" / f"hemi_{hemisphere}"
    source_labels_path = subject_dir / "individual_labels_k3.npy"
    source_valid_path = (
        shared_surface_root
        / subject
        / "surface"
        / "tsnr_gate"
        / f"{subject}_hemi-{hemisphere}_valid_mask.npy"
    )
    source_surface_path = _surface_path(
        hippunfold_root, subject, hemisphere, space="corobl", density="512"
    )
    native_surface_path = _surface_path(
        hippunfold_root, subject, hemisphere, space="corobl", density="native"
    )
    source_unfold_surface_path = _surface_path(
        hippunfold_root, subject, hemisphere, space="unfold", density="512"
    )
    native_unfold_surface_path = _surface_path(
        hippunfold_root, subject, hemisphere, space="unfold", density="native"
    )
    native_area_path = _native_area_path(hippunfold_root, subject, hemisphere)

    source_coordinates, _ = load_midthickness(source_surface_path)
    native_coordinates, _ = load_midthickness(native_surface_path)
    source_labels = _load_vector(source_labels_path, dtype=np.int32, label="cross-fit labels")
    source_valid = _load_vector(source_valid_path, dtype=bool, label="source valid mask")
    native_area = _load_gifti_vector(
        native_area_path, dtype=np.float64, label="surfareacorobl metric"
    )
    if source_labels.size != source_coordinates.shape[0] or source_valid.size != source_labels.size:
        raise ValueError(f"HCP source labels/mask do not match the 512 surface: {subject} {hemisphere}")
    if native_area.size != native_coordinates.shape[0]:
        raise ValueError(f"HCP native area does not match the native surface: {subject} {hemisphere}")
    if not np.isfinite(native_area).all() or np.any(native_area < 0):
        raise ValueError(f"HCP native surfareacorobl metric is invalid: {native_area_path}")

    entry = {
        "branch": HCP_METHOD,
        "subject": subject,
        "smoothing_mm": int(smoothing_mm),
        "hemisphere": hemisphere,
        "source_surface_path": source_surface_path,
        "surface_path": native_surface_path,
        "source_unfold_surface_path": source_unfold_surface_path,
        "native_unfold_surface_path": native_unfold_surface_path,
        "vertex_area_mm2": native_area,
        "labels": np.where(source_valid & np.isin(source_labels, (1, 2, 3)), source_labels, 0),
        "valid_mask": source_valid,
    }
    provenance = {
        "subject": subject,
        "hemisphere": hemisphere,
        "smoothing_mm": int(smoothing_mm),
        "template_fold_id": str(combo["template_fold_id"]),
        "subject_was_in_template": False,
        "combo_manifest": str((subject_dir / "combo_manifest.json").resolve()),
        "crossfit_label_path": str(source_labels_path.resolve()),
        "source_valid_mask_path": str(source_valid_path.resolve()),
        "source_surface_path": str(source_surface_path.resolve()),
        "native_surface_path": str(native_surface_path.resolve()),
        "native_area_metric_path": str(native_area_path.resolve()),
    }
    return entry, provenance


def _fingerprint_assets(assets: list[tuple[str, Path]]) -> tuple[str, list[dict[str, str]]]:
    digest = hashlib.sha256()
    records: list[dict[str, str]] = []
    for label, raw_path in sorted(assets, key=lambda item: (item[0], str(item[1]))):
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required HCP input asset is missing: {path}")
        file_sha256 = _sha256(path)
        digest.update(f"{label}\t{path}\t{file_sha256}\n".encode("utf-8"))
        records.append({"label": label, "path": str(path), "sha256": file_sha256})
    return digest.hexdigest(), records


def _input_asset_fingerprint(
    combos: list[tuple[str, str, Path, dict[str, Any]]],
    *,
    crossfit_root: Path,
    shared_surface_root: Path,
    hippunfold_root: Path,
    smoothing_mm: int,
) -> tuple[str, list[dict[str, str]]]:
    assets: list[tuple[str, Path]] = []
    for subject, hemisphere, combo_path, _combo in combos:
        subject_dir = crossfit_root / "subjects" / subject / f"{smoothing_mm}mm" / f"hemi_{hemisphere}"
        assets.extend(
            [
                ("combo_manifest", combo_path),
                ("crossfit_labels", subject_dir / "individual_labels_k3.npy"),
                (
                    "source_valid_mask",
                    shared_surface_root
                    / subject
                    / "surface"
                    / "tsnr_gate"
                    / f"{subject}_hemi-{hemisphere}_valid_mask.npy",
                ),
                ("source_area_surface", _surface_path(hippunfold_root, subject, hemisphere, space="corobl", density="512")),
                ("native_area_surface", _surface_path(hippunfold_root, subject, hemisphere, space="corobl", density="native")),
                ("source_registration_surface", _surface_path(hippunfold_root, subject, hemisphere, space="unfold", density="512")),
                ("native_registration_surface", _surface_path(hippunfold_root, subject, hemisphere, space="unfold", density="native")),
                ("native_area_metric", _native_area_path(hippunfold_root, subject, hemisphere)),
            ]
        )
    return _fingerprint_assets(assets)



def _cache_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "hcp7t_crossfit_area_2mm.tsv", output_dir / "run_manifest.json"


def _load_current_cache(
    output_dir: Path,
    *,
    source_manifest_sha256: str,
    expected_rows: int,
    input_asset_fingerprint: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    feature_path, manifest_path = _cache_paths(output_dir)
    if not feature_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    if (
        manifest.get("cache_schema_version") != _CACHE_SCHEMA_VERSION
        or manifest.get("source_crossfit_manifest_sha256") != source_manifest_sha256
        or manifest.get("feature_rows") != expected_rows
        or manifest.get("input_asset_fingerprint") != input_asset_fingerprint
    ):
        return None
    table = pd.read_csv(feature_path, sep="\t")
    required = {"subject", "cohort", "method", "smoothing_mm", "hemisphere", "system", "surface_area_mm2"}
    if len(table) != expected_rows or not required.issubset(table.columns):
        return None
    return table, manifest


def load_hcp7t_area_features(
    config: Mapping[str, Any] | str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    workbench_command: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load or build the HCP 7T cross-fit native-area reference table."""

    if isinstance(config, (str, Path)):
        config_dict = _load_json(_resolve_repo_path(config))
    elif isinstance(config, Mapping):
        config_dict = dict(config)
    else:
        raise TypeError("HCP reference config must be a mapping or JSON path")
    reference = _reference_spec(config_dict)
    crossfit_root = _resolve_repo_path(str(reference["crossfit_root"]))
    crossfit_manifest_path = _resolve_repo_path(
        str(reference.get("crossfit_manifest", crossfit_root / "crossfit_manifest.json"))
    )
    hippunfold_root = _resolve_repo_path(str(reference["hippunfold_root"]))
    shared_surface_root = _resolve_repo_path(str(reference["shared_surface_root"]))
    smoothing_mm = int(reference.get("smoothing_mm", HCP_SMOOTHING_MM))
    expected_subjects = int(reference.get("expected_n", EXPECTED_HCP_SUBJECTS))
    if expected_subjects != EXPECTED_HCP_SUBJECTS:
        raise ValueError("HCP 7T reference configuration must specify 151 subjects")
    if smoothing_mm != HCP_SMOOTHING_MM:
        raise ValueError("HCP 7T reference is defined only for 2 mm smoothing")
    run_manifest = validate_crossfit_run_manifest(_load_json(crossfit_manifest_path))
    subjects = _subject_list(run_manifest)
    combos = _combo_specs(crossfit_root, subjects, smoothing_mm=smoothing_mm)
    input_asset_fingerprint, input_assets = _input_asset_fingerprint(
        combos,
        crossfit_root=crossfit_root,
        shared_surface_root=shared_surface_root,
        hippunfold_root=hippunfold_root,
        smoothing_mm=smoothing_mm,
    )
    source_manifest_sha256 = _sha256(crossfit_manifest_path)
    destination = Path(output_dir).expanduser().resolve()
    expected_rows = EXPECTED_HCP_SUBJECTS * len(HEMISPHERES) * len(SYSTEMS)
    cached = _load_current_cache(
        destination,
        source_manifest_sha256=source_manifest_sha256,
        expected_rows=expected_rows,
        input_asset_fingerprint=input_asset_fingerprint,
    )
    if cached is not None:
        return cached

    destination.mkdir(parents=True, exist_ok=True)
    workbench = _resolve_workbench(workbench_command)
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for subject, hemisphere, _combo_path, combo in combos:
        entry, combo_provenance = _build_hcp_entry(
            subject=subject,
            hemisphere=hemisphere,
            smoothing_mm=smoothing_mm,
            combo=combo,
            crossfit_root=crossfit_root,
            hippunfold_root=hippunfold_root,
            shared_surface_root=shared_surface_root,
        )
        native_labels, native_valid, transfer = align_labels_to_native(
            entry,
            output_dir=destination,
            workbench_command=workbench,
        )
        area = compute_system_surface_area(
            entry["vertex_area_mm2"], native_labels, valid_mask=native_valid
        )
        aligned_label_path = (
            destination
            / "label_alignment"
            / HCP_METHOD
            / subject
            / f"{smoothing_mm}mm"
            / f"hemi-{hemisphere}"
            / "native_labels.label.gii"
        )
        combo_provenance.update(
            {
                "native_labels_path": str(aligned_label_path.resolve()),
                "label_transfer_method": transfer["method"],
                "native_valid_vertex_count": int(np.sum(native_valid)),
                "native_labeled_vertex_count": int(area["valid_labeled_vertex_count"]),
                "vertex_area_source": area["vertex_area_source"],
            }
        )
        provenance.append(combo_provenance)
        for system in SYSTEMS:
            rows.append(
                {
                    "subject": subject,
                    "cohort": HCP_GROUP,
                    "method": HCP_METHOD,
                    "smoothing_mm": smoothing_mm,
                    "hemisphere": hemisphere,
                    "system": system,
                    "status": "success",
                    "surface_area_mm2": float(area["systems"][system]["surface_area_mm2"]),
                    "template_fold_id": str(combo["template_fold_id"]),
                    "subject_was_in_template": False,
                    "crossfit_combo_manifest": str((_combo_path).resolve()),
                    "crossfit_label_path": str(
                        (crossfit_root / "subjects" / subject / f"{smoothing_mm}mm" / f"hemi_{hemisphere}" / "individual_labels_k3.npy").resolve()
                    ),
                }
            )

    table = pd.DataFrame(rows)
    feature_path, manifest_path = _cache_paths(destination)
    table.to_csv(feature_path, sep="\t", index=False)
    output_manifest: dict[str, Any] = {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "status": "ok",
        "cohort": HCP_GROUP,
        "method": HCP_METHOD,
        "smoothing_mm": smoothing_mm,
        "n_subjects": len(subjects),
        "n_folds": EXPECTED_HCP_FOLDS,
        "family_aware": True,
        "subject_only_waiver": False,
        "source_crossfit_manifest": str(crossfit_manifest_path.resolve()),
        "source_crossfit_manifest_sha256": source_manifest_sha256,
        "input_asset_fingerprint": input_asset_fingerprint,
        "input_asset_count": len(input_assets),
        "input_assets": input_assets,
        "native_area_source": "hippunfold_native_surfareacorobl",
        "native_area_convention": "native per-vertex surfareacorobl values summed by ADAP_BARY_AREA-transferred held-out cross-fit labels",
        "feature_rows": len(table),
        "outputs": {"features": str(feature_path.resolve())},
        "combination_provenance": provenance,
    }
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table, output_manifest


def _finite_values(table: pd.DataFrame, *, cohort: str, hemisphere: str, system: str) -> np.ndarray:
    values = pd.to_numeric(
        table.loc[
            table["cohort"].eq(cohort)
            & table["hemisphere"].eq(hemisphere)
            & table["system"].eq(system),
            "surface_area_mm2",
        ],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _pooled_sd(case: np.ndarray, reference: np.ndarray) -> float:
    denominator = case.size + reference.size - 2
    if denominator <= 0:
        return float("nan")
    variance = (
        (case.size - 1) * np.var(case, ddof=1)
        + (reference.size - 1) * np.var(reference, ddof=1)
    ) / denominator
    return float(np.sqrt(variance)) if variance > 0 else float("nan")


def compute_hcp7t_reference_effect(
    hcp_features: pd.DataFrame,
    nthc_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive HCP 7T minus Stanford NTHC area effects."""

    required = {"subject", "cohort", "hemisphere", "system", "surface_area_mm2"}
    for table, label in ((hcp_features, "HCP feature table"), (nthc_features, "NTHC feature table")):
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for hemisphere in HEMISPHERES:
        for system in SYSTEMS:
            case = _finite_values(hcp_features, cohort=HCP_GROUP, hemisphere=hemisphere, system=system)
            reference = _finite_values(nthc_features, cohort="NTHC", hemisphere=hemisphere, system=system)
            if case.size < 2 or reference.size < 2:
                raise ValueError(f"HCP/NTHC reference effect needs at least two values: {hemisphere} {system}")
            difference = float(np.mean(case) - np.mean(reference))
            test = scipy_stats.ttest_ind(case, reference, equal_var=False, nan_policy="omit")
            case_variance = float(np.var(case, ddof=1))
            reference_variance = float(np.var(reference, ddof=1))
            standard_error = float(np.sqrt(case_variance / case.size + reference_variance / reference.size))
            welch_df = float(
                (case_variance / case.size + reference_variance / reference.size) ** 2
                / (
                    (case_variance / case.size) ** 2 / (case.size - 1)
                    + (reference_variance / reference.size) ** 2 / (reference.size - 1)
                )
            )
            critical = float(scipy_stats.t.ppf(0.975, welch_df))
            pooled = _pooled_sd(case, reference)
            rows.append(
                {
                    "contrast": "HCP 7T−NTHC",
                    "case_group": HCP_GROUP,
                    "reference_group": "NTHC",
                    "metric_family": "surface_area",
                    "metric": "surface_area_mm2",
                    "source_field": "surface_area_mm2",
                    "method": HCP_METHOD,
                    "smoothing_mm": HCP_SMOOTHING_MM,
                    "hemisphere": hemisphere,
                    "system": system,
                    "case_n": int(case.size),
                    "case_mean": float(np.mean(case)),
                    "case_sd": float(np.std(case, ddof=1)),
                    "reference_n": int(reference.size),
                    "reference_mean": float(np.mean(reference)),
                    "reference_sd": float(np.std(reference, ddof=1)),
                    "mean_difference": difference,
                    "welch_t": float(test.statistic),
                    "welch_df": welch_df,
                    "p_value": float(test.pvalue),
                    "p_raw": float(test.pvalue),
                    "cohens_d": difference / pooled if np.isfinite(pooled) else float("nan"),
                    "ci_low": difference - critical * standard_error,
                    "ci_high": difference + critical * standard_error,
                    "case_missing_n": 0,
                    "reference_missing_n": 0,
                    "case_missing_subjects": "",
                    "reference_missing_subjects": "",
                    "missing_reason": "",
                    "comparison_scope": "descriptive cross-dataset reference; excluded from Stanford FDR family",
                }
            )
    return pd.DataFrame(rows)

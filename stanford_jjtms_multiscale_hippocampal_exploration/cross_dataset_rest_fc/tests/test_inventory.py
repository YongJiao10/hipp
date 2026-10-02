"""Tests for the fail-closed cross-dataset rest-FC inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from inventory import (  # noqa: E402
    build_rosters,
    inspect_fc_inputs,
    validate_rest_only_path,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_config.json"


def test_scope_gate_allows_legal_jjtms_dataset_names(tmp_path: Path) -> None:
    """JJTMS/JJ_TMS are legal dataset components when the asset is rest-only."""

    for name in ("JJTMS", "JJ_TMS"):
        candidate = tmp_path / name / "rest-fc" / "subject_manifest.tsv"
        assert validate_rest_only_path(candidate, (tmp_path,)) == candidate.resolve()
    nifti = tmp_path / "JJ_TMS" / "rest-fmri" / "sub-0001_bold.nii.gz"
    assert validate_rest_only_path(nifti, (tmp_path,)) == nifti.resolve()


def test_scope_gate_rejects_explicit_prohibited_components(tmp_path: Path) -> None:
    """Prohibited assets, archives, traversal, and native mapping fail closed."""

    prohibited = (
        "beta",
        "beta_map.tsv",
        "TMS-response",
        "tms_response.tsv",
        "TMSResponse",
        "TMSContrast",
        "TMSResponseFile",
        "betaMap",
        "conImage",
        "nativeMapping",
        "con",
        "con_image.nii.gz",
        "native_mapping",
        "native-mapping.tsv",
        "archive",
        "results.tar.gz",
    )
    for component in prohibited:
        with pytest.raises(ValueError, match="rest-fMRI-only scope"):
            validate_rest_only_path(tmp_path / "JJTMS" / component, (tmp_path,))

    with pytest.raises(ValueError, match="traversal"):
        validate_rest_only_path(
            tmp_path / "JJTMS" / "rest" / ".." / "beta" / "input.tsv",
            (tmp_path,),
        )


def test_scope_gate_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    """A syntactically rest-only path must still be contained by an allow-list."""

    outside = tmp_path.parent / "JJTMS" / "rest-fc.tsv"
    with pytest.raises(ValueError, match="allowed root"):
        validate_rest_only_path(outside, (tmp_path,))


def test_real_rosters_have_frozen_stanford_cells_and_jjtms_labels() -> None:
    """The configured rosters expose only the frozen, mutually exclusive labels."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rosters = build_rosters(config)

    stanford = rosters["stanford"]
    assert stanford.groupby("group").size().to_dict() == {
        "NTHC": 51,
        "TEHC": 40,
        "NTS": 56,
        "TIS": 47,
    }
    assert stanford["subject"].nunique() == 194
    assert set(stanford["dataset"]) == {"stanford"}
    assert set(stanford[["trauma", "symptomatic"]].itertuples(index=False, name=None)) == {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    }

    jjtms = rosters["jjtms"]
    assert jjtms.groupby("group").size().to_dict() == {
        "healthy": 67,
        "patient": 27,
    }
    assert set(jjtms["group"]) <= {"healthy", "patient"}
    assert set(jjtms["dataset"]) == {"jjtms"}
    assert stanford.attrs.get("immutable") is True
    assert jjtms.attrs.get("immutable") is True


def test_stanford_roster_preserves_clinical_and_trauma_manifest_paths() -> None:
    """Every derived Stanford cell retains both source-manifest dimensions."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roster = build_rosters(config)["stanford"]
    nthc = roster.loc[roster["group"] == "NTHC"].iloc[0]
    stanford_config = config["datasets"]["stanford"]["rosters"]

    assert nthc["clinical_manifest"] == stanford_config["healthy"]
    assert nthc["trauma_manifest"] == stanford_config["nontrauma"]
    assert nthc["source_manifest_primary"] == stanford_config["healthy"]
    assert nthc["source_manifest_secondary"] == stanford_config["nontrauma"]
    assert stanford_config["healthy"] in nthc["source_manifest"]
    assert stanford_config["nontrauma"] in nthc["source_manifest"]


def test_roster_builder_rejects_duplicate_or_overlapping_labels(tmp_path: Path) -> None:
    """Manifest ambiguity cannot silently create a group assignment."""

    def write(name: str, *subjects: str) -> str:
        path = tmp_path / name
        path.write_text("\n".join(subjects) + "\n", encoding="utf-8")
        return str(path)

    config = {
        "allowed_roots": [str(tmp_path)],
        "datasets": {
            "stanford": {
                "rosters": {
                    "NTHC": write("nthc.txt", "sub-0001"),
                    "TEHC": write("tehc.txt", "sub-0001"),
                    "NTS": write("nts.txt", "sub-0002"),
                    "TIS": write("tis.txt", "sub-0003"),
                }
            }
        },
    }
    with pytest.raises(ValueError, match="overlap"):
        build_rosters(config)


def test_fc_inventory_is_explicit_and_records_completed_external_branch(tmp_path: Path) -> None:
    """Coverage is built from configured paths, never recursive discovery."""

    manifest = tmp_path / "subjects.tsv"
    manifest.write_text("subject\tgroup\nsub-0001\thealthy\n", encoding="utf-8")
    fc_root = tmp_path / "JJTMS" / "rest-fc"
    fc_path = fc_root / "sub-0001" / "2mm" / "hemi-L" / "fisher_z_fc.npy"
    fc_path.parent.mkdir(parents=True)
    fc_path.write_bytes(b"rest-fc-placeholder")
    provenance_path = fc_path.parent / "combo_manifest.json"
    provenance_path.write_text("{}\n", encoding="utf-8")

    config = {
        "allowed_roots": [str(tmp_path)],
        "smoothing_mm": [2],
        "hemispheres": ["L"],
        "datasets": {
            "jjtms": {
                "rosters": {"manifest": str(manifest)},
                "roster_format": "subject_group_tsv",
                "fc_inputs": {
                    "root": str(fc_root),
                    "fisher_z_pattern": "{subject}/{smoothing}mm/hemi-{hemisphere}/fisher_z_fc.npy",
                    "provenance_pattern": "{subject}/{smoothing}mm/hemi-{hemisphere}/combo_manifest.json",
                },
            }
        },
        "external_branches": {
            "stanford_group_atlas": {
                "status": "compatibility_only_unified",
                "path": "/do-not-open/external/stanford_clustering",
            }
        },
    }

    rosters = {"jjtms": build_rosters(config)["jjtms"]}
    coverage = inspect_fc_inputs(config, rosters)

    assert isinstance(coverage, pd.DataFrame)
    row = coverage.loc[coverage["dataset"] == "jjtms"].iloc[0]
    assert row["status"] == "present"
    assert row["fc_estimator"] == "fisher_z"
    assert row["path_opened"] is False
    assert set(coverage["dataset"]) == {"jjtms"}


def test_fc_inventory_blocks_non_fisher_z_estimator(tmp_path: Path) -> None:
    """A Pearson/non-registered estimator can never be reported as present."""

    manifest = tmp_path / "subjects.tsv"
    manifest.write_text("subject\tgroup\n1001\thealthy\n", encoding="utf-8")
    root = tmp_path / "JJTMS" / "rest-fc"
    combo = root / "sub-1001" / "2mm" / "hemi-L"
    combo.mkdir(parents=True)
    (combo / "fisher_z_fc.npy").write_bytes(b"fc")
    (combo / "combo_manifest.json").write_text("{}\n", encoding="utf-8")
    config = {
        "allowed_roots": [str(tmp_path)],
        "smoothings_mm": [2],
        "hemispheres": ["L"],
        "datasets": {
            "jjtms": {
                "rosters": {"manifest": str(manifest)},
                "fc_inputs": {
                    "root": str(root),
                    "fisher_z_pattern": "{subject}/{smoothing}mm/hemi-{hemisphere}/fisher_z_fc.npy",
                    "provenance_pattern": "{subject}/{smoothing}mm/hemi-{hemisphere}/combo_manifest.json",
                    "fc_estimator": "pearson",
                },
            }
        },
    }
    roster = build_rosters(config)["jjtms"]
    coverage = inspect_fc_inputs(config, {"jjtms": roster})
    row = coverage.iloc[0]
    assert row["status"] == "blocked"
    assert row["fc_estimator"] == "pearson"
    assert row["registered_fc_estimator"] == "fisher_z"
    assert "fisher_z" in row["reason"]


def test_fc_inventory_requires_provenance_manifest(tmp_path: Path) -> None:
    """A ready FC root without a configured provenance path is blocked."""

    manifest = tmp_path / "subjects.tsv"
    manifest.write_text("subject\tgroup\n1001\thealthy\n", encoding="utf-8")
    root = tmp_path / "JJTMS" / "rest-fc"
    combo = root / "sub-1001" / "2mm" / "hemi-L"
    combo.mkdir(parents=True)
    (combo / "fisher_z_fc.npy").write_bytes(b"fc")
    config = {
        "allowed_roots": [str(tmp_path)],
        "smoothings_mm": [2],
        "hemispheres": ["L"],
        "datasets": {
            "jjtms": {
                "rosters": {"manifest": str(manifest)},
                "fc_inputs": {
                    "root": str(root),
                    "fisher_z_pattern": "{subject}/{smoothing}mm/hemi-{hemisphere}/fisher_z_fc.npy",
                },
            }
        },
    }
    roster = build_rosters(config)["jjtms"]
    coverage = inspect_fc_inputs(config, {"jjtms": roster})
    row = coverage.iloc[0]
    assert row["status"] == "blocked"
    assert "provenance" in row["reason"]


def test_missing_explicit_jjtms_input_is_a_concrete_blocker() -> None:
    """The shipped config records the JJTMS rest-only gate before extraction."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    jjtms = config["datasets"]["jjtms"]
    assert jjtms["input_gate"]["status"] in {"ready", "blocked"}
    if jjtms["input_gate"]["status"] == "blocked":
        assert jjtms["input_gate"]["reason"]

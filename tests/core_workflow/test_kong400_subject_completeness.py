from __future__ import annotations

import importlib
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "core_workflow"))

atlas = importlib.import_module("scripts.core_workflow.05_02_run_group_atlas")
fc_contract = importlib.import_module("scripts.core_workflow.lib.kong2022_fc_contract")


@pytest.mark.parametrize("route_tag", ["routea", "routea_ctxreg", "routea_ctxdrop"])
def test_production_kong_annotation_call_consumes_controlled_profile_timeseries(tmp_path, route_tag):
    # Execute the real CLI call expression with real annotation inputs, avoiding
    # unrelated bootstrap/render stages. Wrong root reverses the observed FC.
    original = tmp_path / "original"
    controlled = tmp_path / "controlled"
    reference = tmp_path / "reference"
    increasing = np.tile(np.arange(5, dtype=np.float32), (4, 1))
    # Opposing vertex noise cancels in the cluster mean. Averaging vertex FC
    # instead produces a much smaller statistic despite identical inputs.
    noise = np.asarray([12, -9, 7, -13, 3], dtype=np.float32)
    increasing[::2] += noise
    increasing[1::2] -= noise
    for subject in ("01", "02"):
        _write_kong_annotation_subject(original, reference, subject, ts=-increasing)
        _write_kong_annotation_subject(controlled, reference, subject, ts=increasing)
    tree = ast.parse(Path(atlas.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "annotate_kong2022_clusters_topn"]
    assert len(calls) == 1
    scope = dict(vars(atlas))
    scope.update(
        labels_final=np.array([1, 1, 2, 2]), support=np.ones(4, bool),
        subjects=["01", "02"], shared_root=tmp_path, surface_root=original,
        config=SimpleNamespace(profile_surface_root=controlled),
        reference_root=reference, smoothing="2mm", hemi="L",
        valid_masks={s: np.ones(4, bool) for s in ("01", "02")},
        network_names=atlas.canonical_families_7_order(), network_taxonomy="canonical7",
        atlas="kong2022", profile_source_mode="canonical",
        route_tag=route_tag,
        profile_source_root=None, profile_source_registry=None,
    )
    result = eval(compile(ast.Expression(calls[0]), atlas.__file__, "eval"), scope)
    assert result[0].shape == (2, 7)
    assert np.all(result[0] > 0), "Annotation used the anticorrelated original TS instead of the configured control"
    expected = atlas.fisher_z_transform_fc(
        atlas.corrcoef_rows(increasing[:2].mean(axis=0, keepdims=True),
                            np.arange(5, dtype=np.float32)[None, :])
    )[0, 0]
    np.testing.assert_allclose(result[0], expected, rtol=1e-5)
    assert result[5] == ["01", "02"]


def test_vertex_fc_fisherz_is_rejected_as_an_unsupported_profile_mode():
    with pytest.raises(ValueError, match="expected 'cluster_mean_timeseries'"):
        atlas.cluster_roi_profile_subjectwise(
            group_labels_final=np.array([1], dtype=np.int32),
            support_mask=np.ones(1, dtype=bool),
            subjects=[],
            shared_root=Path("/does/not/exist"),
            surface_root=Path("/does/not/exist"),
            reference_root=Path("/does/not/exist"),
            smoothing="2mm",
            hemi="L",
            valid_masks_by_subject={},
            profile_fc_mode="vertex_fc_fisherz",
        )


def _write_network_reference_subject(
    surface_root: Path,
    reference_root: Path,
    subject: str,
    *,
    atlas_name: str = "fixture_atlas",
    ts: np.ndarray | None = None,
    network_ts: np.ndarray | None = None,
    write_surface: bool = True,
    write_reference: bool = True,
) -> tuple[Path, Path, Path]:
    ts_path = (
        surface_root
        / f"sub-{subject}"
        / "surface"
        / "2mm"
        / f"sub-{subject}_hemi-L_timeseries.npy"
    )
    ref_dir = reference_root / f"sub-{subject}" / "reference" / atlas_name
    order_path = ref_dir / "cortex_family17_networks.tsv"
    network_path = ref_dir / "cortex_family17_network_timeseries.npy"
    if write_surface:
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            ts_path,
            np.asarray(
                ts
                if ts is not None
                else [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0],
                      [5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
                dtype=np.float32,
            ),
        )
    if write_reference:
        ref_dir.mkdir(parents=True, exist_ok=True)
        order_path.write_text(
            "family17_network\nDefault\nVisual\n",
            encoding="utf-8",
        )
        np.save(
            network_path,
            np.asarray(
                network_ts
                if network_ts is not None
                else [[5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
                dtype=np.float32,
            ),
        )
        (ref_dir / "reference_summary.json").write_text(json.dumps({
            "run_ids": [1], "run_lengths": [np.load(network_path).shape[-1]],
        }))
    return ts_path, order_path, network_path


def _cluster_network_profile_kwargs(surface_root: Path, reference_root: Path) -> dict[str, object]:
    return {
        "group_labels_final": np.asarray([1, 1, 2, 2], dtype=np.int32),
        "support_mask": np.ones(4, dtype=bool),
        "subjects": ["01", "02"],
        "shared_root": reference_root / "unused_shared",
        "surface_root": surface_root,
        "reference_root": reference_root,
        "atlas": "fixture_atlas",
        "smoothing": "2mm",
        "hemi": "L",
        "valid_masks_by_subject": {
            "01": np.ones(4, dtype=bool),
            "02": np.ones(4, dtype=bool),
        },
        "profile_fc_mode": "pearson",
        "network_taxonomy": "family17",
    }


@pytest.mark.parametrize(
    "failure_kind",
    [
        "missing_surface",
        "missing_reference",
        "surface_not_2d",
        "reference_not_2d",
        "frame_mismatch",
        "missing_valid_mask",
        "valid_mask_shape_mismatch",
    ],
)
def test_cluster_network_profile_fails_closed_for_incomplete_requested_subject(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    surface_root = tmp_path / "surface"
    reference_root = tmp_path / "reference"
    _write_network_reference_subject(surface_root, reference_root, "01")

    kwargs = _cluster_network_profile_kwargs(surface_root, reference_root)
    kwargs["subjects"] = ["01", "02"]
    if failure_kind == "missing_surface":
        _write_network_reference_subject(
            surface_root, reference_root, "02", write_surface=False
        )
    elif failure_kind == "missing_reference":
        _write_network_reference_subject(
            surface_root, reference_root, "02", write_reference=False
        )
    elif failure_kind == "surface_not_2d":
        _write_network_reference_subject(
            surface_root,
            reference_root,
            "02",
            ts=np.arange(4, dtype=np.float32),
        )
    elif failure_kind == "reference_not_2d":
        _write_network_reference_subject(
            surface_root,
            reference_root,
            "02",
            network_ts=np.arange(5, dtype=np.float32),
        )
    elif failure_kind == "frame_mismatch":
        _write_network_reference_subject(
            surface_root,
            reference_root,
            "02",
            network_ts=np.arange(12, dtype=np.float32).reshape(2, 6),
        )
    elif failure_kind == "missing_valid_mask":
        _write_network_reference_subject(surface_root, reference_root, "02")
        kwargs["valid_masks_by_subject"] = {"01": np.ones(4, dtype=bool)}
    elif failure_kind == "valid_mask_shape_mismatch":
        _write_network_reference_subject(surface_root, reference_root, "02")
        kwargs["valid_masks_by_subject"]["02"] = np.ones(3, dtype=bool)
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(failure_kind)

    with pytest.raises((FileNotFoundError, ValueError, KeyError), match=r"sub-02|02") as exc_info:
        atlas.cluster_network_profile(**kwargs)

    message = str(exc_info.value)
    if failure_kind in {"missing_surface", "missing_reference"}:
        assert "sub-02" in message
        assert str(tmp_path) in message
    else:
        assert "sub-02" in message
        assert "mismatch" in message.lower() or "2d" in message.lower() or "valid mask" in message.lower()


def test_cluster_network_profile_keeps_empty_supported_cluster_as_valid_output(tmp_path: Path) -> None:
    surface_root = tmp_path / "surface"
    reference_root = tmp_path / "reference"
    _write_network_reference_subject(surface_root, reference_root, "01")
    kwargs = _cluster_network_profile_kwargs(surface_root, reference_root)
    kwargs["subjects"] = ["01"]
    kwargs["valid_masks_by_subject"] = {"01": np.ones(4, dtype=bool)}
    kwargs["support_mask"] = np.asarray([True, True, False, False], dtype=bool)

    profile = atlas.cluster_network_profile(**kwargs)

    assert profile.shape == (2, 2)


def _write_kong_registry(path: Path) -> None:
    rows = fc_contract.authoritative_kong2022_roi_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["parcel_id", "parcel_name", "hemisphere", "family17", "subregion", "rank", "canonical8", "canonical7"]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[({"parcel_id": "roi_id", "parcel_name": "roi_name", "hemisphere": "hemi"}).get(field, field)]) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_kong_annotation_subject(
    surface_root: Path,
    reference_root: Path,
    subject: str,
    *,
    ts: np.ndarray | None = None,
    roi_ts: np.ndarray | None = None,
    write_surface: bool = True,
    write_reference: bool = True,
) -> tuple[Path, Path, Path]:
    ts_path = (
        surface_root
        / f"sub-{subject}"
        / "surface"
        / "2mm"
        / f"sub-{subject}_hemi-L_timeseries.npy"
    )
    ref_dir = reference_root / f"sub-{subject}" / "reference" / "kong2022"
    roi_path = ref_dir / "cortex_roi_parcel_timeseries.npy"
    registry_path = ref_dir / "cortex_roi_parcels.tsv"
    if write_surface:
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            ts_path,
            np.asarray(
                ts
                if ts is not None
                else [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0],
                      [5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
                dtype=np.float32,
            ),
        )
    if write_reference:
        ref_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            roi_path,
            np.asarray(
                roi_ts
                if roi_ts is not None
                else np.tile(np.arange(5, dtype=np.float32), (400, 1)),
                dtype=np.float32,
            ),
        )
        _write_kong_registry(registry_path)
        (ref_dir / "reference_summary.json").write_text(json.dumps({
            "run_ids": [1], "run_lengths": [np.load(roi_path).shape[-1]],
        }))
    return ts_path, roi_path, registry_path


def _kong_annotation_kwargs(surface_root: Path, reference_root: Path) -> dict[str, object]:
    return {
        "group_labels_final": np.asarray([1, 1, 2, 2], dtype=np.int32),
        "support_mask": np.ones(4, dtype=bool),
        "subjects": ["01", "02"],
        "shared_root": reference_root / "unused_shared",
        "surface_root": surface_root,
        "reference_root": reference_root,
        "smoothing": "2mm",
        "hemi": "L",
        "valid_masks_by_subject": {
            "01": np.ones(4, dtype=bool),
            "02": np.ones(4, dtype=bool),
        },
        "network_names": ["Default"],
        "q_thresh": 0.05,
        "top_n_cap": 20,
        "network_column": "canonical7",
        "reference_atlas": "kong2022",
    }


@pytest.mark.parametrize(
    "failure_kind",
    [
        "missing_surface",
        "missing_reference",
        "surface_not_2d",
        "reference_not_2d",
        "frame_mismatch",
        "missing_valid_mask",
        "valid_mask_shape_mismatch",
    ],
)
def test_kong_annotation_fails_closed_without_shrinking_requested_roster(
    tmp_path: Path,
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = fc_contract.authoritative_kong2022_roi_registry()
    monkeypatch.setattr(atlas, "load_kong2022_roi_registry", lambda: registry)
    surface_root = tmp_path / "surface"
    reference_root = tmp_path / "reference"
    _write_kong_annotation_subject(surface_root, reference_root, "01")

    kwargs = _kong_annotation_kwargs(surface_root, reference_root)
    kwargs["network_names"] = ["Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language"]
    if failure_kind == "missing_surface":
        _write_kong_annotation_subject(
            surface_root, reference_root, "02", write_surface=False
        )
    elif failure_kind == "missing_reference":
        _write_kong_annotation_subject(
            surface_root, reference_root, "02", write_reference=False
        )
    elif failure_kind == "surface_not_2d":
        _write_kong_annotation_subject(
            surface_root,
            reference_root,
            "02",
            ts=np.arange(4, dtype=np.float32),
        )
    elif failure_kind == "reference_not_2d":
        _write_kong_annotation_subject(
            surface_root,
            reference_root,
            "02",
            roi_ts=np.arange(400, dtype=np.float32),
        )
    elif failure_kind == "frame_mismatch":
        _write_kong_annotation_subject(
            surface_root,
            reference_root,
            "02",
            roi_ts=np.arange(400 * 6, dtype=np.float32).reshape(400, 6),
        )
    elif failure_kind == "missing_valid_mask":
        _write_kong_annotation_subject(surface_root, reference_root, "02")
        kwargs["valid_masks_by_subject"] = {"01": np.ones(4, dtype=bool)}
    elif failure_kind == "valid_mask_shape_mismatch":
        _write_kong_annotation_subject(surface_root, reference_root, "02")
        kwargs["valid_masks_by_subject"]["02"] = np.ones(3, dtype=bool)
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(failure_kind)

    with pytest.raises((FileNotFoundError, ValueError, KeyError), match="sub-02") as exc_info:
        atlas.annotate_kong2022_clusters_topn(**kwargs)

    message = str(exc_info.value)
    assert "sub-02" in message
    if failure_kind in {"missing_surface", "missing_reference"}:
        assert str(tmp_path) in message
    else:
        assert "mismatch" in message.lower() or "2d" in message.lower() or "valid mask" in message.lower()

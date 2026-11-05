from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.visualize.cortex_network_area.analysis import (
    COMPARISONS,
    apply_grouped_bh_fdr,
    build_statistics_table,
    build_subject_records,
    compute_welch_summary,
    flatten_subject_metrics,
    expand_variants,
    run_analysis,
    write_analysis_outputs,
)
from scripts.visualize.cortex_network_area.core import (
    AreaMetrics,
    InputLayout,
    SubjectRecord,
    Taxonomy,
)


def test_welch_summary_reports_case_minus_control_effects() -> None:
    summary = compute_welch_summary([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])

    assert summary["n_case"] == 3
    assert summary["n_control"] == 3
    assert summary["mean_case"] == pytest.approx(5.0)
    assert summary["mean_control"] == pytest.approx(2.0)
    assert summary["difference_pp"] == pytest.approx(3.0)
    assert summary["relative_difference_percent"] == pytest.approx(150.0)
    assert summary["ci95_low_pp"] == pytest.approx(0.7330421)
    assert summary["ci95_high_pp"] == pytest.approx(5.2669579)
    assert summary["cohens_d"] == pytest.approx(3.0)
    assert summary["p_value"] == pytest.approx(0.02131164)


def _subject_metrics() -> pd.DataFrame:
    cohort_values = {
        "HCP HC": [9.0, 10.0, 11.0],
        "Iowa HC": [10.0, 11.0, 12.0],
        "Patient": [12.0, 13.0, 14.0],
        "Pure MDD": [13.0, 14.0, 15.0],
        "Comorbid MDD": [11.0, 12.0, 13.0],
    }
    rows: list[dict[str, object]] = []
    for cohort, values in cohort_values.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "subject": f"{cohort}-{index}",
                    "cohort": cohort,
                    "variant": "36p",
                    "method": "kong2022",
                    "network_classification": "canonical7",
                    "taxonomy": "kong_canonical7",
                    "network": "Default",
                    "component_kind": "network",
                    "area_percent": value,
                    "area_mm2": value * 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_statistics_table_contains_all_seven_fixed_comparisons() -> None:
    statistics = build_statistics_table(_subject_metrics())

    assert len(COMPARISONS) == 7
    assert statistics["comparison"].tolist() == [item.key for item in COMPARISONS]
    assert set(statistics["n_case"]) == {3}
    assert set(statistics["n_control"]) == {3}
    patient_iowa = statistics.loc[
        statistics["comparison"] == "patient_vs_iowa_hc"
    ].iloc[0]
    assert patient_iowa["case_cohort"] == "Patient"
    assert patient_iowa["control_cohort"] == "Iowa HC"
    assert patient_iowa["difference_pp"] == pytest.approx(2.0)
    assert patient_iowa["difference_mm2"] == pytest.approx(20.0)


def test_bh_fdr_groups_across_variants_comparisons_and_networks() -> None:
    statistics = pd.DataFrame(
        {
            "method": ["kong2022", "kong2022", "kong2022", "lynch2024"],
            "network_classification": ["canonical7"] * 4,
            "variant": ["36p", "36p", "36p", "36p"],
            "comparison": ["a", "b", "c", "a"],
            "network": ["Default", "Visual", "Control", "Default"],
            "p_value": [0.01, 0.04, 0.03, 0.01],
        }
    )

    corrected = apply_grouped_bh_fdr(statistics)

    assert corrected.loc[:2, "q_value_bh"].tolist() == pytest.approx([0.03, 0.04, 0.04])
    assert corrected.loc[3, "q_value_bh"] == pytest.approx(0.01)
    assert corrected["fdr_family_size"].tolist() == [3, 3, 3, 1]


def test_output_writer_exports_full_tables_audit_and_manifest(tmp_path: Path) -> None:
    metrics = _subject_metrics().iloc[:2].copy()
    statistics = build_statistics_table(_subject_metrics())
    audit = pd.DataFrame(
        [{"subject": "sub-1001", "status": "ready", "missing_inputs": ""}]
    )
    manifest = {
        "variant_request": "36p",
        "check_readiness": False,
        "actual_n": {"36p": {"Iowa HC": 3}},
    }

    paths = write_analysis_outputs(
        tmp_path,
        subject_metrics=metrics,
        statistics=statistics,
        input_audit=audit,
        run_manifest=manifest,
    )

    assert set(paths) == {
        "subject_metrics",
        "statistics",
        "input_audit",
        "run_manifest",
    }
    assert pd.read_csv(paths["subject_metrics"]).shape == metrics.shape
    assert pd.read_csv(paths["statistics"]).shape == statistics.shape
    assert pd.read_csv(paths["input_audit"]).loc[0, "status"] == "ready"
    assert json.loads(paths["run_manifest"].read_text(encoding="utf-8")) == manifest


def test_calculation_failure_is_fail_closed_and_audit_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.visualize.cortex_network_area import core

    record = SubjectRecord(
        subject="sub-corrupt",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={"L": tmp_path / "L.surf.gii", "R": tmp_path / "R.surf.gii"},
        label_path=tmp_path / "labels.txt",
        functional_manifest=tmp_path / "source.json",
    )
    monkeypatch.setattr(
        core,
        "audit_subject_records",
        lambda records: [
            {
                "subject": record.subject,
                "cohort": record.cohort,
                "dataset": record.dataset,
                "variant": record.variant,
                "atlas": record.atlas,
                "status": "ready",
                "missing_inputs": "",
                "error": "",
            }
        ],
    )

    def fail_calculation(*args, **kwargs):
        raise ValueError("corrupt labels")

    monkeypatch.setattr(core, "calculate_subject_record", fail_calculation)
    output_dir = tmp_path / "stats"
    with pytest.raises(RuntimeError, match="incompatible"):
        run_analysis(
            repo_root=Path(__file__).resolve().parents[2],
            output_dir=output_dir,
            variant="36p",
            records=[record],
        )

    audit = pd.read_csv(output_dir / "input_audit.csv")
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert audit.loc[0, "status"] == "incompatible"
    assert "corrupt labels" in audit.loc[0, "error"]
    assert pd.read_csv(output_dir / "subject_network_area_metrics.csv").empty
    assert pd.read_csv(output_dir / "network_area_statistics.csv").empty
    assert manifest["analysis_status"] == "failed_input_compatibility"


def test_all_variant_expands_in_fixed_preprocessing_order() -> None:
    assert expand_variants("36p") == ("36p",)
    assert expand_variants("all") == ("36p",)
    with pytest.raises(ValueError, match="36p.*all"):
        expand_variants("2mm")


def test_readiness_mode_exports_missing_inputs_without_calculating(
    tmp_path: Path,
) -> None:
    record = SubjectRecord(
        subject="sub-1001",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={
            "L": tmp_path / "missing.L.surf.gii",
            "R": tmp_path / "missing.R.surf.gii",
        },
        label_path=tmp_path / "missing.dlabel.nii",
        functional_manifest=tmp_path / "missing.provenance.json",
    )

    result = run_analysis(
        repo_root=Path(__file__).resolve().parents[2],
        output_dir=tmp_path / "stats",
        variant="36p",
        check_readiness=True,
        records=[record],
    )

    audit = pd.read_csv(result["input_audit"])
    manifest = json.loads(result["run_manifest"].read_text(encoding="utf-8"))
    assert audit.loc[0, "status"] == "missing"
    assert pd.read_csv(result["subject_metrics"]).empty
    assert pd.read_csv(result["statistics"]).empty
    assert manifest["variant_request"] == "36p"
    assert manifest["check_readiness"] is True
    assert manifest["audit_status_counts"] == {"missing": 1}


def _layout(tmp_path: Path) -> InputLayout:
    return InputLayout(
        hcp_surface_template=str(tmp_path / "hcp/{sub_subject}/{hemi}.surf.gii"),
        hcp_functional_manifest_template=str(tmp_path / "hcp/ica_fix.json"),
        hcp_label_templates={
            "kong2022": str(tmp_path / "hcp/{sub_subject}/kong"),
            "lynch2024": str(tmp_path / "hcp/{sub_subject}/lynch.dlabel.nii"),
        },
        iowa_surface_templates={
            "36p": str(tmp_path / "iowa/{sub_subject}/{hemi}.surf.gii"),
        },
        iowa_functional_manifest_templates={
            "36p": str(tmp_path / "iowa36/{sub_subject}/provenance.json"),
        },
        iowa_label_templates={
            "36p": {
                "kong2022": str(tmp_path / "iowa36/{sub_subject}/kong"),
                "lynch2024": str(tmp_path / "iowa36/{sub_subject}/lynch.dlabel.nii"),
            },
        },
    )


def test_subject_record_builder_uses_every_formal_roster_for_both_methods(
    tmp_path: Path,
) -> None:
    records = build_subject_records(
        repo_root=Path(__file__).resolve().parents[2],
        variants=("36p",),
        layout=_layout(tmp_path),
    )

    assert len(records) == (151 + 67 + 26 + 7 + 19) * 2
    assert {record.atlas for record in records} == {"kong2022", "lynch2024"}
    assert {record.cohort for record in records} == {
        "HCP HC",
        "Iowa HC",
        "Patient",
        "Pure MDD",
        "Comorbid MDD",
    }
    assert all(record.subject != "sub-3003" for record in records)


def test_flattened_metrics_keep_network_noise_and_unassigned_separate() -> None:
    record = SubjectRecord(
        subject="sub-1001",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={"L": Path("L.surf.gii"), "R": Path("R.surf.gii")},
        label_path=Path("labels.dlabel.nii"),
        functional_manifest=Path("source.json"),
    )
    taxonomy = Taxonomy(
        key="kong_canonical7",
        method="kong2022",
        classification="canonical7",
        network_order=("Default", "Visual"),
        label_to_network={},
        source="config/cross_atlas_network_merge.json",
    )
    metrics = AreaMetrics(
        network_area_mm2={"Default": 60.0, "Visual": 20.0},
        area_percent={"Default": 60.0, "Visual": 20.0},
        total_cortical_area_mm2=100.0,
        noise_area_mm2=5.0,
        unassigned_area_mm2=15.0,
    )

    rows = flatten_subject_metrics(
        record, {"kong_canonical7": metrics}, {"kong_canonical7": taxonomy}
    )

    assert [row["network"] for row in rows] == [
        "Default",
        "Visual",
        "Noise",
        "Unassigned",
    ]
    assert [row["component_kind"] for row in rows] == [
        "network",
        "network",
        "noise",
        "unassigned",
    ]
    assert rows[-2]["area_percent"] == pytest.approx(5.0)
    assert rows[-1]["area_percent"] == pytest.approx(15.0)
    assert rows[0]["mapping_source"] == "config/cross_atlas_network_merge.json"




def test_full_run_rejects_missing_inputs_before_partial_statistics(tmp_path):
    from scripts.visualize.cortex_network_area.core import SubjectRecord
    from scripts.visualize.cortex_network_area.analysis import run_analysis
    import pytest
    record = SubjectRecord('sub-1001', 'Iowa HC', 'iowa', '36p', 'lynch2024',
                           {'L': tmp_path/'missing.L.surf.gii', 'R': tmp_path/'missing.R.surf.gii'},
                           tmp_path/'missing.dlabel.nii', tmp_path/'missing.json')
    with pytest.raises(RuntimeError, match='missing or incompatible'):
        run_analysis(repo_root=Path(__file__).resolve().parents[2], output_dir=tmp_path/'stats', variant='36p', records=[record])
    assert pd.read_csv(tmp_path/'stats'/'network_area_statistics.csv').empty

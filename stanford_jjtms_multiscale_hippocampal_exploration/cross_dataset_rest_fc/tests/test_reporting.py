"""Contract tests for Task 6 deterministic reporting and evidence packaging.

The fixture is intentionally small, but it exercises the complete reporting
schema: both datasets, both resolutions, all localisation estimands, all
hemisphere modes, and both available methods.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import contrasts  # noqa: E402
import plotting  # noqa: E402
import reporting  # noqa: E402


METHODS = ("hcp_fixed", "hcp_template_matching")
REGISTERED_METHODS = METHODS + ("group_atlas_gradient_spectral",)
MODES = ("bilateral", "L", "R", "L-R")
REGIONS = ("anterior", "middle", "posterior")


def test_run_timestamp_is_current_utc_not_historical_snapshot() -> None:
    started = datetime.now(timezone.utc)
    value = reporting._run_timestamp()
    finished = datetime.now(timezone.utc)

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert started <= parsed <= finished
    assert value != "2026-08-24T00:00:00Z"


def _primary_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, estimands in (
        ("jjtms", contrasts.JJTMS_ESTIMANDS),
        ("stanford", contrasts.STANFORD_ESTIMANDS),
    ):
        for estimand in estimands:
            for method in METHODS:
                for smoothing in (2, 4):
                    for mode in MODES:
                        for resolution, networks in ((8, ("Default",)), (17, ("DefaultA",))):
                            for region in REGIONS:
                                for network in networks:
                                    coefficient = 0.25 if dataset == "jjtms" else -0.4
                                    rows.append(
                                        {
                                            "dataset": dataset,
                                            "estimand": estimand,
                                            "resolution": resolution,
                                            "method": method,
                                            "smoothing_mm": smoothing,
                                            "hemisphere_mode": mode,
                                            "region": region,
                                            "network": network,
                                            "fc_estimator": "fisher_z",
                                            "n": 6,
                                            "coefficient": coefficient,
                                            "hc3_se": 0.1,
                                            "ci_low": coefficient - 0.2,
                                            "ci_high": coefficient + 0.2,
                                            "standardized_effect": coefficient,
                                            "p_raw": 0.01,
                                            "q_bh": 0.04,
                                            "correction_status": "deferred_until_analysis_set_freeze",
                                            "failure_reason": "",
                                        }
                                    )
    return reporting.add_source_row_ids(pd.DataFrame(rows), "primary")


def _descriptives_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for resolution, network in (
        (8, plotting.NETWORKS_8[0]),
        (17, plotting.NETWORKS_17[0]),
    ):
        for group_index, group in enumerate(contrasts.STANFORD_CELLS):
            rows.append(
                {
                    "method": "hcp_fixed",
                    "smoothing_mm": 2,
                    "hemisphere_mode": "bilateral",
                    "network_resolution": resolution,
                    "region": "anterior",
                    "network": network,
                    "dataset": "stanford",
                    "group": group,
                    "n": 8,
                    "mean": 0.02 * (group_index + 1),
                    "sd": 0.1,
                    "ci_low": -0.03,
                    "ci_high": 0.07,
                    "units": "fisher_z",
                    "status": "present",
                    "failure_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_primary_contract_contains_both_datasets_resolutions_estimands_modes_and_methods() -> None:
    frame = _primary_fixture()
    assert reporting.REGISTERED_METHODS == REGISTERED_METHODS
    assert set(frame["dataset"]) == {"jjtms", "stanford"}
    assert set(frame["resolution"]) == {8, 17}
    assert set(frame["hemisphere_mode"]) == set(MODES)
    assert set(frame["method"]) == set(METHODS)
    assert set(frame["fc_estimator"]) == {"fisher_z"}
    assert set(frame.loc[frame.dataset.eq("jjtms"), "estimand"]) == {"patient-healthy"}
    assert set(frame.loc[frame.dataset.eq("stanford"), "estimand"]) == set(
        contrasts.STANFORD_ESTIMANDS
    )
    assert frame["source_row_id"].is_unique
    assert "coefficient" in frame.columns


def test_unified_branch_manifest_keeps_group_atlas_in_priority_plan() -> None:
    config = {
        "unified_branch_policy": {
            "method_order": [
                "hcp_fixed",
                "hcp_template_matching",
                "group_atlas_gradient_spectral",
            ],
            "subject_fc_estimands": {
                "jjtms": ["patient-healthy"],
                "stanford": ["symptom_marginal"],
            },
            "subject_fc_priority": [
                {"tier": 0, "network_resolution": 8, "smoothing_mm": 2},
                {"tier": 1, "network_resolution": 8, "smoothing_mm": 4},
                {"tier": 1, "network_resolution": 17, "smoothing_mm": 2},
                {"tier": 2, "network_resolution": 17, "smoothing_mm": 4},
            ],
            "group_atlas_priority": [
                {"tier": 0, "smoothing_mm": 2},
                {"tier": 2, "smoothing_mm": 4},
            ],
            "barrier": "strict",
        },
        "datasets": {"jjtms": {}, "stanford": {}},
        "smoothings_mm": [2, 4],
        "hemispheres": ["L", "R"],
        "group_atlas": {
            "method": "group_atlas_gradient_spectral",
            "reference_group": "HC",
            "groups": ["HC", "NTHC"],
            "assignment_variants": ["plain"],
            "smoothings_mm": [2, 4],
            "hemispheres": ["L", "R"],
        },
    }
    plan = reporting.build_branch_plan(config)
    calls: list[str] = []

    def execute(branch):
        calls.append(branch.branch_id)
        return "complete"

    _, _, statuses, manifest = reporting.run_priority_batches(
        config,
        {},
        plan,
        execute,
    )
    assert all(status == "complete" for status in statuses.values())
    assert any(
        row["analysis_family"] == "group_atlas_geometry"
        and row["method"] == "group_atlas_gradient_spectral"
        for row in manifest["branches"]
    )
    group_tiers = {
        row["smoothing_mm"]: row["priority_tier"]
        for row in manifest["branches"]
        if row["analysis_family"] == "group_atlas_geometry"
    }
    assert group_tiers[2] == 0
    assert group_tiers[4] == 2
    assert len(calls) == len(plan)


def test_candidate_registry_is_small_and_untested_without_exhaustive_pairs() -> None:
    candidates = reporting.build_candidate_registry(_primary_fixture())
    assert len(candidates) <= 5
    assert "p_pair" not in candidates.columns
    assert "q_pair" not in candidates.columns
    assert "source_row_id_a" in candidates.columns
    assert "source_row_id_b" in candidates.columns
    assert "selection_rule" in candidates.columns
    assert "future_validation_target" in candidates.columns
    assert "pair_count" not in candidates.columns


def test_candidate_registry_deduplicates_pairs_before_applying_cap() -> None:
    rows = []
    effects = {"A": 0.01, "B": -0.03, "C": 0.02}
    for method in ("hcp_fixed", "hcp_template_matching"):
        for smoothing_mm in (2, 4):
            for network, coefficient in effects.items():
                if (
                    network == "A"
                    and method == "hcp_template_matching"
                    and smoothing_mm == 4
                ):
                    coefficient = -coefficient
                rows.append(
                    {
                        "dataset": "stanford",
                        "estimand": "interaction",
                        "resolution": 8,
                        "hemisphere_mode": "bilateral",
                        "region": "posterior",
                        "network": network,
                        "method": method,
                        "smoothing_mm": smoothing_mm,
                        "fc_estimator": "fisher_z",
                        "coefficient": coefficient,
                    }
                )
    candidates = reporting.build_candidate_registry(pd.DataFrame(rows), cap=2)
    assert len(candidates) == 2
    assert candidates["candidate_id"].is_unique
    pairs = list(zip(candidates["network_a"], candidates["network_b"]))
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {("B", "C"), ("B", "A")}
    b_a = candidates.loc[
        candidates["network_a"].eq("B") & candidates["network_b"].eq("A")
    ].iloc[0]
    assert b_a["cross_spec_sign_consistency_a"] == pytest.approx(1.0)
    assert b_a["cross_spec_sign_consistency_b"] == pytest.approx(0.75)


def test_direction_consistency_is_limited_to_aligned_comparisons() -> None:
    table = reporting.build_direction_consistency(_primary_fixture())
    assert set(table["stanford_estimand"]) <= {
        "symptom_marginal",
        "NTS-NTHC",
        "TIS-TEHC",
    }
    assert set(table["jjtms_estimand"]) == {"patient-healthy"}
    assert "source_row_id_jjtms" in table.columns
    assert "source_row_id_stanford" in table.columns


def test_artifact_is_compact_with_visible_scope_notice_and_safe_sources(tmp_path: Path) -> None:
    frame = _primary_fixture().head(24)
    bundle = {
        "primary_localisation": frame,
        "group_counts": pd.DataFrame(
            [{"dataset": "jjtms", "group": "healthy", "n": 1}]
        ),
        "auxiliary_profiles": pd.DataFrame(
            [{"dataset": "jjtms", "p_perm": 0.5, "n_perm": 499}]
        ),
        "candidate_registry": reporting.build_candidate_registry(_primary_fixture()),
        "direction_consistency": reporting.build_direction_consistency(_primary_fixture()),
        "negative_findings": pd.DataFrame(
            [{"finding_type": "external_separate", "method": "seven_cohort_group_atlas"}]
        ),
    }
    artifact = reporting.build_artifact(bundle, run_id="test_run")
    assert artifact["surface"] == "report"
    assert artifact["manifest"]["title"] == artifact["manifest"]["blocks"][0]["body"].split("\n", 1)[0].lstrip("# ")
    assert [block["id"] for block in artifact["manifest"]["blocks"]] == [
        "title",
        "primary_chart",
    ]
    assert all(
        block.get("layout") == "full"
        for block in artifact["manifest"]["blocks"]
    )
    body = "\n".join(
        block.get("body", "")
        for block in artifact["manifest"]["blocks"]
        if block.get("type") == "markdown"
    )
    assert "Stanford group-atlas geometry is executed by the same unified priority plan" in body
    assert "technical_report.md" in body
    assert artifact["snapshot"]["status"] == "ready"
    assert artifact["snapshot"]["accessIssues"] == []
    assert [chart["id"] for chart in artifact["manifest"]["charts"]] == [
        "primary_effect_ci"
    ]
    assert artifact["manifest"]["tables"] == []
    assert set(artifact["snapshot"]["datasets"]) == {"primary_effect_ci"}

    technical_report = reporting._technical_report(
        "test_run",
        {"primary_localisation": frame},
        {},
        bundle["candidate_registry"],
    )
    for section in (
        "Technical summary",
        "Key findings",
        "Scope, data, and metric definitions",
        "Methodology",
        "Limitations",
        "Recommended next steps",
        "Further questions",
    ):
        assert section in technical_report
    for source in artifact["sources"]:
        assert not str(source.get("path", "")).startswith("/")
        assert ".." not in Path(str(source.get("path", ""))).parts


def test_artifact_accepts_real_descriptive_network_resolution_schema() -> None:
    rows = reporting._profile_snapshot(_descriptives_fixture()).to_dict("records")
    assert len(rows) == 8
    assert {row["network_resolution"] for row in rows} == {8, 17}
    assert all("x_index" in row and "y_index" in row for row in rows)


def test_artifact_direction_table_is_bounded_for_portable_layout() -> None:
    snapshot = reporting._direction_snapshot(
        reporting.build_direction_consistency(_primary_fixture())
    )
    fields = list(snapshot.columns)
    assert fields == [
        "comparison",
        "method",
        "hemisphere_mode",
        "resolution",
        "region",
        "network",
        "jjtms_coefficient",
        "stanford_coefficient",
        "direction_relation",
    ]
    rows = snapshot.to_dict("records")
    assert rows
    assert set(rows[0]) == set(fields)


def test_portable_chart_snapshots_are_bounded_without_losing_both_resolutions() -> None:
    heatmap = reporting._heatmap_snapshot(_primary_fixture())
    assert set(heatmap["method"]) == {"hcp_fixed"}
    assert set(heatmap["resolution"]) == {8, 17}
    assert "failure_reason" not in heatmap.columns
    assert set(heatmap.columns) == {
        "source_row_id",
        "dataset",
        "estimand",
        "resolution",
        "method",
        "smoothing_mm",
        "hemisphere_mode",
        "region",
        "network",
        "n",
        "coefficient",
        "p_raw",
        "correction_status",
        "x_index",
        "y_index",
        "branch",
    }

    profiles = reporting._profile_snapshot(_descriptives_fixture())
    assert set(profiles["network_resolution"]) == {8, 17}
    assert "failure_reason" not in profiles.columns
    assert set(profiles.columns) == {
        "method",
        "smoothing_mm",
        "hemisphere_mode",
        "network_resolution",
        "region",
        "network",
        "dataset",
        "group",
        "n",
        "mean",
        "ci_low",
        "ci_high",
        "x_index",
        "y_index",
    }

    auxiliary = pd.DataFrame(
        [
            {
                "dataset": "jjtms",
                "estimand": "patient-healthy",
                "method": "hcp_fixed",
                "smoothing_mm": 2,
                "hemisphere_mode": "bilateral",
                "network_resolution": 8,
                "fc_estimator": "fisher_z",
                "profile_scope": "subregion_profile",
                "region": "anterior",
                "n_total": 94,
                "n_complete": 90,
                "effective_dimension": 8,
                "statistic": 1.5,
                "p_perm": 0.04,
                "n_perm": 499,
                "derived_seed": 123,
                "failure_reason": "x" * 2000,
            }
        ]
    )
    auxiliary_snapshot = reporting._auxiliary_snapshot(auxiliary)
    assert "failure_reason" not in auxiliary_snapshot.columns
    assert auxiliary_snapshot.iloc[0]["profile_status"] == "failed"


def test_specification_snapshot_has_real_x_field_and_bounded_labels() -> None:
    consistency = reporting.build_specification_consistency(_primary_fixture())
    rows = reporting._consistency_snapshot(consistency).to_dict("records")
    assert rows
    assert {row["spec_index"] for row in rows} <= {0.0, 1.0, 2.0, 3.0}
    assert all("component_label" in row for row in rows)


def test_colorbar_figures_are_warning_free(tmp_path: Path) -> None:
    primary_info = plotting.plot_primary_localisation_heatmaps(
        _primary_fixture(), tmp_path, run_id="warning_free"
    )
    descriptive_info = plotting.plot_stanford_four_group_profiles(
        _descriptives_fixture(), tmp_path, run_id="warning_free"
    )
    assert Path(primary_info["png"]).is_file()
    assert Path(descriptive_info["png"]).is_file()
    assert "pdf" not in primary_info
    assert "pdf" not in descriptive_info
    assert not (tmp_path / "primary_localisation_heatmaps.pdf").exists()
    assert not (tmp_path / "stanford_four_group_profiles.pdf").exists()
    plotting.plot_primary_localisation_heatmaps(
        _primary_fixture(), tmp_path, run_id="warning_free"
    )
    plotting.plot_stanford_four_group_profiles(
        _descriptives_fixture(), tmp_path, run_id="warning_free"
    )


def test_current_writer_is_deterministic_and_overwrites_existing_table(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"b": 2, "a": 1}])
    first = reporting.write_table(frame, tmp_path / "table.csv.gz", source_row_id_prefix="x")
    second_bytes = first.read_bytes()
    repeated = reporting.write_table(frame, tmp_path / "table_repeat.csv.gz", source_row_id_prefix="x")
    assert second_bytes == repeated.read_bytes()
    replacement = pd.DataFrame([{"b": 3, "a": 1}])
    reporting.write_table(replacement, first, source_row_id_prefix="x")
    assert first.read_bytes() != second_bytes

def test_technical_report_does_not_present_provisional_q_values_as_final() -> None:
    primary = _primary_fixture()
    primary["q_bh"] = np.nan
    primary["correction_status"] = "deferred_until_analysis_set_freeze"
    text = reporting._technical_report(
        "deferred",
        {"primary_localisation": primary},
        {},
        pd.DataFrame(),
        seed=20260903,
    )
    assert "multiplicity correction is deferred until the final analysis-object set is frozen" in text.lower()
    assert "seed=20260903" in text
    assert "seed=20260824" not in text
    assert "q < .05" not in text
    assert "q < .10" not in text


def test_current_reporting_run_id_is_unversioned() -> None:
    assert reporting.DEFAULT_RUN_ID == "primary_screen"


def test_builder_receipt_history_is_json_safe_after_two_reader_timeouts() -> None:
    timeout_receipt = {
        "ok": False,
        "stage": "static_charts",
        "code": "reader_timeout",
    }
    finalized = reporting._finalize_builder_receipt(
        timeout_receipt,
        controller_attempt=2,
        prior_reader_timeouts=[timeout_receipt, timeout_receipt],
    )
    serialized = json.dumps(finalized, allow_nan=False)
    assert '"controller_attempt": 2' in serialized
    assert len(finalized["prior_reader_timeouts"]) == 2
    assert all(item is not finalized for item in finalized["prior_reader_timeouts"])

def test_portable_report_builder_tracks_installed_plugin_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = tmp_path / "data-analytics"
    old_dir = cache_root / "0.2.8"
    current_dir = cache_root / "0.2.9"
    builder = current_dir / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
    builder.parent.mkdir(parents=True)
    builder.write_text("// current builder", encoding="utf-8")
    monkeypatch.setattr(reporting, "PORTABLE_REPORT_CACHE_ROOT", cache_root)
    assert reporting._portable_report_builder() == builder.resolve()


def test_missing_html_builder_does_not_capture_stale_report_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()
    figures_dir.mkdir()
    reports_dir.mkdir()
    stale_html = reports_dir / "report.html"
    stale_html.write_text("old report", encoding="utf-8")
    technical = reports_dir / "technical_report.md"
    technical.write_text("current report", encoding="utf-8")

    outputs = reporting._collect_run_outputs(
        results_dir,
        figures_dir,
        reports_dir,
        html_generated=False,
        builder_receipt_written=False,
    )

    assert technical in outputs
    assert stale_html not in outputs
